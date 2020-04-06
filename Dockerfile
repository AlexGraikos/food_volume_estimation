FROM python:3.6

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install food-volume-estimation package
ADD food_volume_estimation/ food_volume_estimation/
copy setup.py .
RUN python setup.py install

# Add model files to image
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.json models/depth_architecture.json
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.h5 models/depth_weights.h5
COPY models/segmentation/mask_rcnn_food_segmentation.h5 models/segmentation_weights.h5

# Copy and execute server script
COPY food_volume_estimation_app.py .
ENTRYPOINT ["python", "food_volume_estimation_app.py", \
            "--depth_model_architecture", "models/depth_architecture.json", \
            "--depth_model_weights", "models/depth_weights.h5", \
            "--segmentation_model_weights", "models/segmentation_weights.h5"]

