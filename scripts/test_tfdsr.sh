python test_tfdsr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--seesr_model_path preset/models/checkpoint-600 \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/data/test \
--output_dir preset/result \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512