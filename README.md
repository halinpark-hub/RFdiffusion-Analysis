# RFdiffusion-example_outputs
RFdiffusion/examples/example_outputs 

## 1. An unconditional monomer 
조건 없이 단백질 구조 생성

../RFdiffusion/examples 에 있는 여러 예시 shell script 파일 중에 unconditional monomer 파일은 총 3개

**I. design_unconditional.sh**

../scripts/run_inference.py inference.output_prefix=example_outputs/design_unconditional 'contigmap.contigs=[100-200]' inference.num_designs=10
가장 기본적인 디자인 실행이며 potential 을 사용하지 않는다.

모델이 자신이 기억하는 학습된 구조 데이터를 바탕으로 구조 생성한다. contact potential 이나 ROG potential과 같은 조건을 살짝 걸어주는 역할이 없다.

**II. design_unconditional_w_contact_potential.sh**

../scripts/run_inference.py inference.output_prefix=example_outputs/design_unconditional_w_contact_potential
'contigmap.contigs=[100-200]' inference.num_designs=10 'potentials.guiding_potentials=["type:monomer_contacts,weight:0.05"]'
기본적인 디자인 코드에 potentials.guiding_potentials 가 추가되어서 type= monomer_contacts, weight :0.05 로 설정한다.

단백질 구조 예측 및 디자인에서 특정 아미노산 잔기들이 서로 얼마나 가까이 있는지를 평가하는 함수이다.

contact potential의 가중치는 높을수록 접촉을 더 강하게 유도한다. (가중치 범위는 0.00 - 1.0+)

해당 모델의 contact potential의 가중치는 0.05로 낮은 편에 속함. 따라서, contact potential이 디자인을 생성하는 데에 있어서 접촉을 강하게 유도하지 않는다.

하지만, 단백질이 무조건 루즈하게 접히는 것은 아니다. 접촉 말고 다른 요소(서열, 폴딩 에너지)가 단백질 접힘에 영향을 줄 수 있기 때문이다.

**III. design_unconditional_w_monomer_ROG.sh**

../scripts/run_inference.py inference.output_prefix=example_outputs/design_monomer_ROG_unconditional
 'contigmap.contigs=[100-200]' inference.num_designs=10 'potentials.guiding_potentials=["type:monomer_ROG,weight:1,min_dist:5"]' potentials.guide_scale=2 potentials.guide_decay="quadratic"
ROG: Radius of Gyration (회전반경) 단백질 구조의 무게중심으로부터, atom들이 얼마나 퍼져 있는지를 나타내는 평균 거리

단백질 구조가 너무 퍼지거나 너무 압축되지 않도록 유지하게 도와주는 potentia
