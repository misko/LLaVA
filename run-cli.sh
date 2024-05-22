#!/bin/bash
set -eu

# Note: other servers are needed too; see:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/LoRA.md

python -m llava.eval.run_llava \
    --model-path llava-v1.5-7b-meals-task-lora \
    --model-base ../llava-v1.5-7b \
    --image-file "https://storage.googleapis.com/rx-food-mnt-prod/meal_photo_resized/81a2e16df5a93be19d636ebcdc289b737237?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wolever-storage-reader%40rx-food-prod.iam.gserviceaccount.com%2F20240521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240521T201807Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&x-goog-signature=7c67830281813820c553cfa2f3d9eae514373b93c751483380f38787b35fff03138fbc381d99064fd480946d13d16798a21cf60968dad861cd7fcdb2d5c6949aa35a675a8b964aa8306cbf42e801c70535d18469624f23e23a8fa5d98f76e0e491791332011953b6c197c5dfdd3ac6885e4dfc92daf40b9e20326ce1ac1d5f861ce0b72087c6b8f2a24bb86b6e4c4dc2160dc4068c43c5b3f45051392b57e38756655176dfe96efee686d99d13b10ca92907b43769a7a6c4e57119e7b3de45f9203650899decb5d2220df37b33667007c438a0eb16ecfe559fa2aaf429763be4e2a3060d90d5893f731a5ea940ba1f95b4093fae8018d6217636d2666f478df1" \
    --query 'A patient has consumed a meal, and has provided a photo of the meal. Describe what the patient ate.'

python -m llava.eval.run_llava \
    --model-path ~/meals-ds-20000-json/output/ \
    --model-base ../llava-v1.5-7b \
    --image-file "https://storage.googleapis.com/rx-food-mnt-prod/meal_photo_resized/81a2e16df5a93be19d636ebcdc289b737237?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wolever-storage-reader%40rx-food-prod.iam.gserviceaccount.com%2F20240521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240521T201807Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&x-goog-signature=7c67830281813820c553cfa2f3d9eae514373b93c751483380f38787b35fff03138fbc381d99064fd480946d13d16798a21cf60968dad861cd7fcdb2d5c6949aa35a675a8b964aa8306cbf42e801c70535d18469624f23e23a8fa5d98f76e0e491791332011953b6c197c5dfdd3ac6885e4dfc92daf40b9e20326ce1ac1d5f861ce0b72087c6b8f2a24bb86b6e4c4dc2160dc4068c43c5b3f45051392b57e38756655176dfe96efee686d99d13b10ca92907b43769a7a6c4e57119e7b3de45f9203650899decb5d2220df37b33667007c438a0eb16ecfe559fa2aaf429763be4e2a3060d90d5893f731a5ea940ba1f95b4093fae8018d6217636d2666f478df1" \
    --query 'A patient has consumed a meal, and has provided a photo of the meal. Describe what the patient ate.'

python -m llava.eval.run_llava \
    --model-path ../llava-v1.5-7b \
    --image-file "https://storage.googleapis.com/rx-food-mnt-prod/meal_photo_resized/81a2e16df5a93be19d636ebcdc289b737237?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wolever-storage-reader%40rx-food-prod.iam.gserviceaccount.com%2F20240521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240521T201807Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&x-goog-signature=7c67830281813820c553cfa2f3d9eae514373b93c751483380f38787b35fff03138fbc381d99064fd480946d13d16798a21cf60968dad861cd7fcdb2d5c6949aa35a675a8b964aa8306cbf42e801c70535d18469624f23e23a8fa5d98f76e0e491791332011953b6c197c5dfdd3ac6885e4dfc92daf40b9e20326ce1ac1d5f861ce0b72087c6b8f2a24bb86b6e4c4dc2160dc4068c43c5b3f45051392b57e38756655176dfe96efee686d99d13b10ca92907b43769a7a6c4e57119e7b3de45f9203650899decb5d2220df37b33667007c438a0eb16ecfe559fa2aaf429763be4e2a3060d90d5893f731a5ea940ba1f95b4093fae8018d6217636d2666f478df1" \
    --query "A patient has consumed a meal, and has provided a photo of the meal. Describe what the patient ate."
