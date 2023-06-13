#!/bin/bash
image=$1 ## 이미지 명 변수 설정
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi
account=$(aws sts get-caller-identity --query Account --output text) ## Account 불러오기
region=$(aws configure get region) ## 리전 명 불러오기
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest" ## ECR 이름 설정

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1 ## 동일한 이름의 ECR 여부 확인

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null ## 없을 경우 새로 생성
fi

$(aws ecr get-login --region ${region} --no-include-email) ## BASE IMAGE 로그인 정보 가져오기
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email) ## BASE IMAGE 로그인 정보 가져오기
base_img='763104351884.dkr.ecr.'$region'.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04' ## BASE IMAGE 설정
echo 'base_img:'$base_img

cd container
docker build -t ${image} -f Dockerfile --build-arg BASE_IMG=$base_img . ## Dockerfile에 있는 내용으로 이미지 Build 실행
docker tag ${image} ${fullname}

docker push ${fullname}
