# RFdiffusion-example_outputs

1. You should always activate conda everytime you run RF diffusion. 

`conda activate rfdiff-cuda113`

SE3nv is saved as rfdiff-cuda113. 


2. Change directory to RFdiffusion
For my RFdiffusion, `cd /halin/RFdiffusion-main/RFdiffusion` is where RFdiffusion is.

---
run_inference.py describes how RF diffusion works. Thus, all examples run by run_inference.py and includes this script. 
**run_inference.py** (RF diffusion Inference script)

```
#!/usr/bin/env python
# python run_inference.py는  hydra 라이브러리를 사용해서 설정 파일 (base.yaml) 로드
# python run_inference.py --config-name symmetry 는 symmetry.yaml 설정 사용
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re  #정규표현식(regex) - 파일명에서 숫자 추출
import os, time, pickle # 파일 처리, 시간 측정, 데이터 저장 및 로드
import torch # 딥러닝 프레임워크, GPU 연산 수행
from omegaconf import OmegaConf #설정파일을 로드
import hydra #설정 파일 관리 라이브러리
import logging # 로그 출력해서 실행 정보를 저장
from rfdiffusion.util import writepdb_multi, writepdb # .pdb 형식의 단백질 구조 데이터 파일 저장 (writepdb_multi, wirtepdb)
from rfdiffusion.inference import utils as iu #sampler_selector 함수 사용해서 샘플러 초기화
from hydra.core.hydra_config import HydraConfig
import numpy as np # 행렬 연산 및 랜덤 시드 설정
import random # 랜덤 시드 설정
import glob # 특정 패턴의 파일을 검색


# 랜덤 시드 고정 함수 (실행의 재현성을 위해 랜덤 시드를 설정)
# 특정 랜덤 시드를 고정하면 매번 같은 값을 도출. 같은 코드를 실행해도 동일한 결과가 나오도록 보장
def make_deterministic(seed=0):
    torch.manual_seed(seed) #pytorch의 랜덤 시드 설정
    np.random.seed(seed) # Numpy 의 랜덤 시드 고정
    random.seed(seed) # Python 기본 random 모듈의 랜덤 시드 설정


# main() 함수
@hydra.main(version_base=None, config_path="../config/inference", config_name="base") #Hydra 를 사용해서 설정 파일 (base.yaml)을 자동으로 로드. conf 변수에 설정 값이 저장됨
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__) # 실행 도중 발생하는 메시지를 기록
    if conf.inference.deterministic: # conf.inference.determinisitc 값이 True 라면 make_deterministic() 호출해서 랜덤 시드 고정
        make_deterministic()

    if torch.cuda.is_available(): # GPU 가 사용가능하면 GPU 실행
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")

    else: # GPU 사용 불가이면 CPU 실행
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # sampler 초기화. (sampler_selector를 사용해서 config.yaml 설정을 기반으로 sampler 생성)
    sampler = iu.sampler_selector(conf)

    # 기존 .pdb 파일이 있는 경우, design_startnum(현재 생성할 디자인 번호(인덱스)) 를 자동으로 증가시켜서 파일 이름에 번호 증가하여 새로 만들어진 디자인이 기존 파일들과 이름이 겹치지 않도록 함.
    if sampler.inf_conf.design_startnum == -1: # design_startnum이 -1이면 기존 .pdb 파일 확인해야 함.
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb") # 기존 파일 검색
        indices = [-1] #초기값 설정: 새로운 디자인을 추가할 때 파일이름의 숫자에 +1을 해서 가장 처음 pdb 파일을 0으로 하고 싶다면 -1로 초기값을 설정해야 함.
        for e in existing: # 파일이 이미 있으면 파일 이름에서 _숫자.pdb 형식의 숫자를 추출하는 정규 표현식을 사용하여 숫자만 추출.
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e) #_숫자.pdb 형식의 숫자 추출 정규 표현식. 예를 들어, output_5.pdb라면 5를 추출
            print(m)
            if not m: # 정규 표현식 매칭 결과가 None 이면 다음 파일을 확인
                continue
            m = m.groups()[0]
            indices.append(int(m)) # 매칭된 숫자만 가져와서 리스트에 추가
        design_startnum = max(indices) + 1 # 가장 큰 번호에 +1을 해서 새로운 디자인 번호 설정


    # 단백질 구조 생성 및 저장
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des) # design_startnum 부터 num_designs(설정된 개수) 만큼 반복하면서 새로운 .pdb 파일 생성. 예를 들어서, design_startnum이 11이고 num_designs가 3이라면 i_des는 11,12,13으로 3번 반복. 또한, deterministic 의 값이 True라면, 랜덤 시드 고정.

        start_time = time.time() # 시작 시간 저장해서 실행 시간 측정
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}" # 생성할 .pdb 파일의 이름 지정
        log.info(f"Making design {out_prefix}") # 어떤 디자인을 생성하는 지 log 를 사용해서 기록
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue # cautious 모드가 켜져 있고, os.path.exists 함수로 같은 이름의 .pdb 파일이 이미 존재하는지 확인되면 해당 디자인을 continue로 건너뜀.


        x_init, seq_init = sampler.sample_init() # sampler.sample_init()을 호출해서 초기 단백질 구조 (x_init) 와 서열(seq_init)을 생성. 이 두개는 Reverse Diffusion 의 input value
        denoised_xyz_stack = [] # denoised된 구조 저장
        px0_xyz_stack = [] # 최종 깨끗한 구조 저장
        seq_stack = [] # 생성된 서열 저장
        plddt_stack = [] # 구조의 신뢰도 점수 저장

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init) # torch.clone 사용해서 원본 데이터를 변경하지 않고 독립적인 사본 생성. diffusion 모델이 원본 데이터 수정하지 않게끔. 원본 데이터를 수정하면 초기값을 사용할 수 없게 되어서 같은 샘플의 재현이 불가능해짐



        # reverse diffusion 수행 (noise 제거해서 단백질 구조 복원 과정)
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1): # range( 초기단계, 최종 단계, -1, -1): t_step.input -> final_step 까지 단계적으로 감소하면서 역방향 확산
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            ) #t: 현재 확산 단계. x_t: 현재 시점의 단백질 구조 좌표, seq_init: 초기 단백질 서열. final_step: 최종 목표 단계
            px0_xyz_stack.append(px0) # 현재 단계에서 예측된 최종 깨끗한 구조 저장
            denoised_xyz_stack.append(x_t) # 현재 t 단계에서 denoise된 구조 저장
            seq_stack.append(seq_t) # 단백질 서열 저장
            plddt_stack.append(plddt[0])  # 신뢰도 점수 저장 (차원 축소 필요)

        # 단백질 구조 좌표 정렬
        denoised_xyz_stack = torch.stack(denoised_xyz_stack) # diffusion 과정에서 실제 사용된 구조 (x_t)를  여러 단계에서 생성된 좌표를 하나의 torch. Tensor로 변환
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0]) #좌표의 순서를 뒤집음 (시간 순서 조정; denoised_xyz_stack은 완전한 노이즈 상태가 시간 상으로 가장 앞에 있음. PyMOL에서 단백질 구조를 시각화할 때, 최종 복원된 구조 먼저 보기 위해서 노이즈가 없는 최종 복원된 구조를 가장 맨 앞에 두기 위해서 flip)
        px0_xyz_stack = torch.stack(px0_xyz_stack) # diffusion 모델이 예측한 최종 깨끗한 구조 (px0)도 마찬가지로 하나의 torch.Tensor로 변환
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0]) # 뒤집기


        # predicted LDDT: 단백질 구조 예측에서 각 원자가 얼마나 신뢰할 수 있는지 나타내는 점수
        plddt_stack = torch.stack(plddt_stack)

        # 각 단계별 단백질 서열 저장.
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1] # -1은 리스트 마지막 요소라는 뜻

        # 특정 모티프(값이 21인 부분)을 제외하고 나머지 부분을 glycine으로 변환
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # torch.where(condition, 7, original). condition(==21) 일 경우에는 유지. 나머지 부분은 Glycine(7)으로 변환.
        # 모티프는 단백질의 핵심 기능을 담당하는 부분이므로 변경하면 안됨. 나머지는 Glycine(아미노산 중에 가장 작고 단순하여 구조적 유연성이 높음)으로 변환하여 서열을 더 쉽게 변경할 수 있게 해줌.

        bfacts = torch.ones_like(final_seq.squeeze()) # B-factor는 원자의 구조적 가변성을 나타냄.
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0 # 모든 위치는 1으로 두고, 특정 아미노산(==21) 은 0으로 둠. 모델이 수정한 특정 위치를 .pdb 파일에서 쉽게 확인할 수 있도록 함.
        out = f"{out_prefix}.pdb"

        # pdb 파일 저장 과정 분석
        writepdb(
            out, #pdb 파일 저장할 경로 (out_prefix.pdb)
            denoised_xyz_stack[0, :, :4], #최종 복원된 단백질 구조의 원자 좌표(X,Y,Z,Atom Type)
            final_seq, #최종 단백질 서열 (seq_stack[-1]에서 추출)
            sampler.binderlen, #특정 단백질 서열이 결합하는 영역의 길이
            chain_idx=sampler.chain_idx, #단백질 체인 정보 (A,B,C 등)
            bfacts=bfacts,# B-factor 값 (각 원자의 구조적 유연성)

        # .trb 파일에 메타데이터 저장
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
        ) # confg: Hydra의 OmegaConf (모델 설정값), plddt: 예측된 신뢰도 점수, device: GPU 또는 CPU

        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value # sampler.contig_map이 존재하면, contiguous mapping (연결된 단백질 구조 정보) 추가.
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out) # .trb 파일로 저장. 실험 결과 분석할 때 사용

        # .pdb Trajectory 파일 저장 (구조 변화 과정)
        if sampler.inf_conf.write_trajectory:
            # write_trajectory = True 이면, diffusion 모델이 구조를 생성하는 과정을 .pdb 파일로 저장
            traj_prefix = (os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix))
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True) #Trajectory 저장 경로 설정

            out = f"{traj_prefix}_Xt-1_traj.pdb" #Xt-1 Trajectory 파일 저장
            writepdb_multi(
                out,
                denoised_xyz_stack, #Diffusion 과정에서 복원된 단백질 구조 좌표
                bfacts,
                final_seq.squeeze(), # 최종 확정된 단백질 서열
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx, #단백질 체인 정보
            )


            out = f"{traj_prefix}_pX0_traj.pdb" #pX0 Trajectory 파일 저장
            writepdb_multi(
                out,
                px0_xyz_stack, #최종 깨끗한 구조
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

        #Xt-1과 pX0 Trajectory 파일을 비교하면 Diffusion 과정에서 구조가 어떻게 변하는 지 확인 가능

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes") #실행 완료 메시지 출력


if __name__ == "__main__":
    main() #스크립트가 직접 실행될 때만 main() 함수를 호출하도록 함.

```

---

# 1. An unconditional monomer 
조건 없이 단백질 구조 생성


../RFdiffusion/examples 에 있는 여러 예시 shell script 파일 중에 unconditional monomer 파일은 총 3개

**I. design_unconditional.sh**
<img width="496" alt="Screenshot 2025-04-20 at 19 46 57" src="https://github.com/user-attachments/assets/de13daed-3b4d-4a22-8dc8-5ec4ba4b2674" />

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_unconditional 'contigmap.contigs=[100-200]' inference.num_designs=10 
```
가장 기본적인 디자인 실행이며 potential 을 사용하지 않는다.

모델이 자신이 기억하는 학습된 구조 데이터를 바탕으로 구조 생성한다. contact potential 이나 ROG potential과 같은 조건을 살짝 걸어주는 역할이 없다.

**II. design_unconditional_w_contact_potential.sh**
<img width="467" alt="Screenshot 2025-04-20 at 19 48 49" src="https://github.com/user-attachments/assets/8b10418b-03c3-4f38-bf9a-f67bad4dfb53" />

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_unconditional_w_contact_potential'contigmap.contigs=[100-200]' inference.num_designs=10 'potentials.guiding_potentials=["type:monomer_contacts,weight:0.05"]'
```

기본적인 디자인 코드에 potentials.guiding_potentials 가 추가되어서 type= monomer_contacts, weight :0.05 로 설정한다.

단백질 구조 예측 및 디자인에서 특정 아미노산 잔기들이 서로 얼마나 가까이 있는지를 평가하는 함수이다.

contact potential의 가중치는 높을수록 접촉을 더 강하게 유도한다. (가중치 범위는 0.00 - 1.0+)

해당 모델의 contact potential의 가중치는 0.05로 낮은 편에 속함. 따라서, contact potential이 디자인을 생성하는 데에 있어서 접촉을 강하게 유도하지 않는다.

하지만, 단백질이 무조건 루즈하게 접히는 것은 아니다. 접촉 말고 다른 요소(서열, 폴딩 에너지)가 단백질 접힘에 영향을 줄 수 있기 때문이다.

**III. design_unconditional_w_monomer_ROG.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_monomer_ROG_unconditional
 'contigmap.contigs=[100-200]' inference.num_designs=10 'potentials.guiding_potentials=["type:monomer_ROG,weight:1,min_dist:5"]' potentials.guide_scale=2 potentials.guide_decay="quadratic"
```
ROG: Radius of Gyration (회전반경) 단백질 구조의 무게중심으로부터, atom들이 얼마나 퍼져 있는지를 나타내는 평균 거리

단백질 구조가 너무 퍼지거나 너무 압축되지 않도록 유지하게 도와주는 potentia


#### design_unconditional monomer.sh - Code Breakdown (All three scripts included)

```
'run_inference.py' #python 스크립트 실행

'inference.output_prefix=example_outputs/design_unconditional_w_contact_potential' # 결과 파일을 저장할 경로(출력 파일=example_outputs/design_unconditional_w_contact_potential 형식으로 저장됨)

'contigmap.contigs = [100-200]' #단백질 서열(contigs) 에서 100~200번 위치를 특정 범위로 설정

'inference.num_designs = 10' #10개의 디자인(구조)를 생성. 결과파일에서 10개의 구조파일이 생성된 것을 볼 수 있음

# addition of potential function
'potentials.guiding_potentials = ["type:monomer_contacts,weights:0.05"]' #contact potential을 적용하여 디자인 수행 (monomer 사용, 가중치 0.05)

#or

'potentials.guiding_potentials = ["type:monomer_ROG,weight:1,min_dist:5"]' # ROG potential을 적용하여 디자인 수행 (monomer 사용, 가중치 1, 최소 거리 5) 가중치 값이 크면 더 강하게 적용됨. 최소 거리 조건에서 ROG 계산
'potentials.guide_scale=2 potentials.guide_decay="quadratic"' # 'scale=2': potential 을 얼마나 강하게 적용할것인지와 'guide_decay=quadratic(2차)': diffusion step이 진행됨에 따라 potential 강도가 2차함수적으로 감소. 초반에는 강하게 potential을 적용하고 점점 자연스럽게 구조를 생성하도록 함
```

---
# 2. Motif Scaffolding

기존에 알려진 단백질의 중요 motif 를 유지하면서, 그 주위에 새로운 단백질 구조를 디자인

motifscaffolding.sh 예시파일은 **총 3개**

**I. design_motifscaffolding.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_motifscaffolding inference.input_pdb=input_pdbs/5TPN.pdb 'contigmap.contigs=[10-40/A163-181/10-40]' inference.num_designs=10
```


**II. design_motifscaffolding_inpanitseq.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_motifscaffolding_inpaintseq inference.input_pdb=input_pdbs/5TPN.pdb 'contigmap.contigs=[10-40/A163-181/10-40]' inference.num_designs=10 'contigmap.inpaint_seq=[A163-168/A170-171/A179]'
```
motif 안에서 몇개만 단백질 기능을 수행하는데 핵심역할을 함.
덜 중요한 residue는 새로 디자인하므로써 단백질이 구조적으로 안정적이게 함.



**III. design_motifscaffolding_with_target.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_motifscaffolding_with_target inference.input_pdb=input_pdbs/1YCR.pdb 'contigmap.contigs=[A25-109/0 0-70/B17-29/0-70]' contigmap.length=70-120 inference.num_designs=10 inference.ckpt_override_path=../models/Complex_base_ckpt.pt
```

residue A25-109: Mdm2 target protein
residue B17-29: p53 helix

p53 단백질의 motif를 고정하고, 앞뒤로 linker를 만듬.
이 linker는 Mdm2 target protein 과 상호작용이 가능한 구조를 scaffolding


complex-finetuned model 사용

___

**design_motifscaffolding.sh- Code Breakdown**

```
../scripts/run_inference.py  #RF diffusion의 구조 생성(inference) 실행하는 파이썬 스크립트

inference.output_prefix=example_outputs/design_motifscaffolding #example_outputs 폴더에 'design_motifscaffolding' 를 생성된 구조가 저장된 파일 이름 접두어로 설정

inference.input_pdb=input_pdbs/5TPN.pdb  # motif가 들어있는 pdb (이 예시에서는 5TPN 단백질 구조. 이 구조 안의 A chain 163-181 을 motif로 사용할 것)

'contigmap.contigs=[10-40/A163-181/10-40]' # A163-181 motf 앞 뒤로 10-40개의 residue 생성

#motifscaffolding_inpaintseq.sh (motif 내부의 일부를 새로 디자인)
contifmap.inpaint_seq = [A163-168/A170-171/A179] # ipaint_seq는 motif 내부에서 sequence만 새로 생성할 residue 를 지정하는 옵션
# A chain 163-168, 170-171, 179만 새로운 sequence 생성


#motifscaffolding_with_target.sh (chain끼리 연결해주는 linker 를 만드는 동시에 target protein과 상호작용할 수 있는 구조 scaffolding)
'contigmap.contigs=[A25-109/0 0-70/B17-29/0-70]'  # chain A 의 25-109번 residue를 고정하고, /0 은 chain A는 끝났다는 의미. 0-70 residue 생성할 공간 (A와 B의 linker) , B chain은 17-29번 고정하고, B chain 뒤로 0-70 residue 새로 생성

contigmap.length=70-120  #모델이 디자인할 전체 구조의 길이 범위. 최소 70- 최대 120개 residue. 고정된 motif도 포함해서 생각해야 함(98residue)
120 - 98 = 22.
#모델이 생성할 수 있는 chain 길이를 최대 120으로 설정했으니 많게는 22 residue를 앞뒤로 생성할 수 있음

inference.ckpt_override_path=../models/Complex_base_ckpt.pt #복합체 구조 생성을 위한 모델 가중치 사용함. (chain A, B를 사용했으므로 )
```

___

## 3. Partial Diffusion


기존 단백질 구조의 일부를 유지하면서, 선택한 영역만을 재설계
partialdiffusion.sh 예시파일은 **총 3개**

**I. design_partialdiffusion.sh**
<img width="537" alt="Screenshot 2025-04-20 at 18 52 52" src="https://github.com/user-attachments/assets/2bdf4b4f-4fc0-41a2-aefc-bb8033ec6248" />
```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_partialdiffusion inference.input_pdb=input_pdbs/2KL8.pdb 'contigmap.contigs=[79-79]' inference.num_designs=10 diffuser.partial_T=10
```

부분적으로만 noise-denoise 하여 원래 구조를 유지하면서 약간의 diversity를 갖는 유사 구조들을 디자인

[1-79]: chain 없는 1에서 79번 residue 위치 (흔하지 않음)
[A1-79] Chain A의 1번 에서 79번 residue 위치
[79-79]: 79 residue 길이

2KL8 단백질의 전체 길이는 79라고 shell script에 적혀 있다. 그래서 contig 을 [79-79] 라고 설정하면 전체를 설정하는 것과 마찬가지.

RF diffusion은 체인없이 [N-N] 숫자만 쓰여서 '길이 N짜리 구조' 라고 해석함

따라서, 구조의 전체를 noise를 준다.

주의해야 할 것은, 입력 pdb 단백질 길이가 79이라면, contigmap.contigs 로 지정하는 길이도 79이여야 한다.

**II. design_partialdiffusion_multipleseq.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_partialdiffusion_peptidewithmultiplesequence inference.input_pdb=input_pdbs/peptide_complex_ideal_helix.pdb 'contigmap.contigs=["172-172/0 34-34"]' diffuser.partial_T=10 inference.num_designs=10 'contigmap.provide_seq=[172-177,200-205]'
```












**III. design_partialdiffusion_withseq.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_partialdiffusion_peptidewithsequence inference.input_pdb=input_pdbs/peptide_complex_ideal_helix.pdb 'contigmap.contigs=["172-172/0 34-34"]' diffuser.partial_T=10 inference.num_designs=10 'contigmap.provide_seq=[172-205]'
```

 **partialdiffusion.sh, multipleseq.sh, withseq.sh - Code Breakdown**

 ```
 #공통 code
 'contigmap.contigs=[79-79]' #partial diffusion 대상 residue 지정. residue 79번 하나만 생성하고 나머지는 고정

 diffuser.partial_T=10 #partial diffusion 의 강도 조절 parameter. 10 step 동안만 79번 residue를 마스킹하고 생성(partial diffusion). 나머지 90step은 refinement.


#multipleseq.sh
'contigmap.provide_seq=[172-177,200-205]' #모델이 172-177, 200-205 residue에 대해서만 알고 있음. 이외의 residue는 masked 상태에서 구조 생성


#withseq.sh
'contigmap.provide_seq=[172-205]' #모델이 172-205 까지 전부 알고 있음. 모델이 훨씬 더 많은 정보를 알기 때문에 정확한 구조 생성에 더 유리하지만 디자인의 자유도는 줄어듦.
```

---
## 4. Binder Design

어떤 단백질 (A) 에 딱 달라붙는 새로운 단백질 (B)를 설계
ppi: protein-protein interaction 기반의 binder 설계
design_ppi.sh 예시파일은 **총 3개**

**I. design_ppi.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_ppi inference.input_pdb=input_pdbs/insulin_target.pdb 'contigmap.contigs=[A1-150/0 70-100]' 'ppi.hotspot_res=[A59,A83,A91]' inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0
```

insulin receptor에 결합하는 binder 단백질을 de novo로 설계.(binder의 topology는 정하지 않아서 자유롭게 만들 수 있도록 함)

타겟 단백질의 구조(150번 잔기)를 기반으로 길이 70-100 아미노산 정도의 새로운 binder를 생성해서 특정 부위 (A59, A83, A91)에 잘 붙도록 설계.
inference 시 노이즈를 0으로 설정해서 더 정밀한 결과를 얻음


**II. design_ppi_flexible_peptide.sh**

```
../scripts/run_inference.py inference.output_prefix=example_outputs/design_ppi_flexible_peptide inference.input_pdb=input_pdbs/3IOL.pdb 'contigmap.contigs=[B10-35/0 70-100]' 'ppi.hotspot_res=[B28,B29]' inference.num_designs=10 'contigmap.inpaint_str=[B10-35]'
```


GLP-1 peptide (chain B, 10~35번)를 topology나 peptide의 구조를 설정해놓지 않고 구조를 유연하게 만들도록 함  
유연한 펩타이드(GLP-1, B10-35번)의 구조를 예측하면서 동시에 그 펩타이드에 결합하는 70-100번 residue 크기의 binder 단백질을 de novo로 설계.



**III. design_ppi_scaffolded.sh**

```
../scripts/run_inference.py scaffoldguided.target_path=input_pdbs/insulin_target.pdb inference.output_prefix=example_outputs/design_ppi_scaffolded scaffoldguided.scaffoldguided=True 'ppi.hotspot_res=[A59,A83,A91]' scaffoldguided.target_pdb=True scaffoldguided.target_ss=target_folds/insulin_target_ss.pt scaffoldguided.target_adj=target_folds/insulin_target_adj.pt scaffoldguided.scaffold_dir=./ppi_scaffolds/ inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0
```

위의 두 예제와는 달리, binder의 scaffold topology (구조적 뼈대)를 미리 정의해서, 그 구조 위에서 binder를 정밀하게 설계.
insulin receptor의 A59, A83, A91 residue 근처에 결합하는 binder를 미리 정의된 단백질 scaffold 구조들 위에서 설계


 **design_ppi.sh- Code Breakdown**

 ```
#design_ppi.sh

'contigmap.contigs=[A1-150/0 70-100]'
# A1-150: target protein. chain A 의 1~150번 residue.
# /0: target 과 binder 사이 간격 없음
#70-100: 길이가 31인 새로 설계할 Binder 단백질. chain을 설정하지 않았으므로 자동으로 지정됨


'ppi.hotspot_res=[A59,A83,A91]' # binder가 결합해야 할 표적 단백질의 표면 '핫스팟' 잔기들
# chain A의 59번, 83번, 91번 residue가 잘 붙도록 유도


denoiser.noise_scale_ca=0, denoiser.noise_scale_frame=0 #noise scale을 0으로 설정했으므로, 구조 변화 없이 거의 정적인 조건에서 디자인. 주로 결합 위치만 고정하고 binder만 새로 설계할 때 사용됨. #프레임(방향,회전)에 적용되는 노이즈 양을 0ㅇ로 설정. 즉, 단백질의 방향기/기울기는 건드리지 않음


#design_ppi_flexible_peptide.sh

contigmap.inpaint_str=[B10-35] #inpainting: 해당 범위를 "비워두고" 모델이 새롭게 설계됨. 즉, 원래 펩타이드 GLP-1의 원래 구조를 그대로 쓰지 않고, RF diffusion이 해당 부위 구조도 함께 예측하게 만듬. 결국에는, binder 설계 + 일부분의 peptide 구조 예측 동시 수행
```

#design_ppi_scaffolded.sh

```
 scaffoldguided.scaffoldguided=True #diffusion을 scaffold-guided 모드로 실행. 즉, 이미 정해진 단백질 구조 뼈대(topology)를 따라서 설계

  scaffoldguided.target_pdb=True #target 구조를 실제 PDB 구조 기반으로 사용할 것이라는 의미

  scaffoldguided.target_ss=target_folds/insulin_target_ss.pt #타겟 단백질의 이차구조 정보
  scaffoldguided.target_adj=target_folds/insulin_target_adj.pt #타겟 단백질의 접근성 정보
  # 이차구조와 접근성 정보는 필수는 아니지만, 명시하면 더 정확한 결합 구조를 생성할 수 있음
```

  scaffoldguided.scaffold_dir=./ppi_scaffolds/ inference. #해당 directory에 다양한 scaffold 구조들 (.pdb)이 저장되어 있음. 모델은 이 중 하나를 골라서 binder의 뼈대로 사용


---

## 5. Fold Conditioning

 **design_ppi_scaffolded.sh- Code Breakdown**

 ## 6. Generation of Symmetric Oligomers
 ---
symmetric oligomers의 예시파일은 **3개 **


**I. design_dihedral_oligos.sh**
<img width="500" alt="Screenshot 2025-04-20 at 18 46 50" src="https://github.com/user-attachments/assets/b5aed332-0599-4288-bf65-1649c2499500" />

D2 대칭을 갖는 oligomer 단백질을 생성.
대칭성을 이용한 de novo 단백질 설계


```
python ../scripts/run_inference.py --config-name=symmetry inference.symmetry="D2" inference.num_designs=10 inference.output_prefix="example_outputs/D2_oligo" 'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2.0 potentials.guide_decay="quadratic" 'contigmap.contigs=[320-320]'
```

총 320개의 아미노산으로 구성된 D2 대칭성 단백질 복합체를 intra- & inter-chain 접촉 유도 potential 을 사용해 10개 설계

 **design_dihedral_oligos.sh- Code Breakdown**

```
--config-name=symmetry #대칭성 설계모드를 활성화
inference.symmetry="D2" #D2 대칭성: 총 4개의 단위체로 구성된 복합체. 2개의 직교 축을 기준으로 회전 대칭성을 가짐


'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' #유도 potential 사용.
#type: oligo_contacts: 단백질 내부 및 복합체 구성 단위체 간의 접촉을 유도.
#weight_intra: 1 동일 chain 내부 접촉은 강하게 유도
#weight_inter:0.1 다른 chain 간 접촉은 약하게 유도

potentials.olig_intra_all=True potentials.olig_inter_all=True #위에서 설정한 intra/inter chain 접촉 potential은 모든 chain에 대해 적용

potentials.guide_scale=2.0 #guiding potential의 scale(세기): 초기값으로 2.0 사용. 이 값이 높을수록 모델이 potential에 더 강하게 따름.

potentials.guide_decay="quadratic" #유도 potential이 점점 줄어드는 방식.

'contigmap.contigs=[320-320]' #총 320개의 아미노산으로 구성도니 단백질 구조를 설계하라는 의미. #D2 대칭이기에, 내부적으로는 80-residue 단위치 4개로 분할됨. (320/4=80)
```

 ---
**II. design_cyclic_oligos.sh**
<img width="500" alt="Screenshot 2025-04-20 at 18 48 41" src="https://github.com/user-attachments/assets/81540f35-2ab2-4345-9e37-0e4f70d147a2" />

C6: 하나의 중심 회전축을 기준으로 60도씩 회전해서 총 6개의 identical subunit이 원형으로 배열된 구조
ex: 바이러스 단백질 capsid, pore 형성 단백질, channel 단백질

```
python ../scripts/run_inference.py --config-name=symmetry inference.symmetry="C6" inference.num_designs=10 inference.output_prefix="example_outputs/C6_oligo" 'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2.0 potentials.guide_decay="quadratic" 'contigmap.contigs=[480-480]'
```



 ---
 **design_cyclic_oligos.sh- Code Breakdown**

```
dihedral_oligos.sh 와 동일한 코드
```

 ---
**III. design_tetrahedral_oligos.sh**
<img width="500" alt="Screenshot 2025-04-20 at 18 50 36" src="https://github.com/user-attachments/assets/4fc14d7a-1b04-4c7e-820b-04282dd13b02" />


정사면체 대칭 (tetrahderal symmetry) 를 가지는 올리고머 단백질을 설계

```
python ../scripts/run_inference.py --config-name=symmetry inference.symmetry="tetrahedral" inference.num_designs=10 inference.output_prefix="example_outputs/tetrahedral_oligo" 'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2.0 potentials.guide_decay="quadratic" 'contigmap.contigs=[1200-1200]'

```

총 1200개의 아미노산으로 구성된, 정사면체 대칭성을 따르는 고차원 대칭성 단백질 복합체를 10개 생성

---

** design_tetrahedral_oligos.sh- Code
Breakdown**

```
dihedral_oligos.sh 와 design_cyclic_oligos.sh 와 동일한 코드
```

## 7. Symmetric Motif Scaffolding


---

**design_nickel.sh**

C4 대칭성을 갖는 단백질 복합체를 설계하면서, 그 안에 대칭적으로 배치된 nickel-binding motif를 정확하게 유지하도록 scaffold(지지 구조)를 만듬

```
python ../scripts/run_inference.py inference.symmetry="C4" inference.num_designs=15 inference.output_prefix=example_outputs/design_nickel 'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.06"]' potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2 potentials.guide_decay="quadratic" inference.input_pdb=input_pdbs/nickel_symmetric_motif.pdb 'contigmap.contigs=[50/A2-4/50/0 50/A7-9/50/0 50/A12-14/50/0 50/A17-19/50/0]' inference.ckpt_override_path=$ckpt
```
nickel-binding motif 4개(A2-4, A7-9,A12-14, A17-19)를 유지한 채
그 주위를 감싸는 C4 대칭의 단백질 scaffold 를 RF diffusion이 설계함

---

**design_nickel.sh- Code
Breakdown**

```
#symmetric oligomers 코드와 동일하고 추가 코드:
inference.ckpt_override_path=$ckpt #성능이 더 좋은 base_epoch8_ckpt.pt 모델을 사용
```
