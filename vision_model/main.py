import cv2
import argparse
from ultralytics import YOLO
import time
import shutil
import os
import supervision as sv
from sentence_transformers import util
from PIL import Image
import asyncio
import aiohttp
import uuid

tracked_badges = {}

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Badge Detector Options')
  parser.add_argument('--name-url', default='http://example.com:8080/v1/models/internvl2:predict', type=str)
  parser.add_argument('--badge-location', default='http://example.com:9696/badge', type=str)
  parser.add_argument('--feed', default=0, type=int)
  parser.add_argument('--resolution', default=[640, 480], nargs=2, type=int)
  args = parser.parse_args()
  return args

async def update_tracked_badges(result, timestamp, name_url, badge_location):
    filename = f'{str(uuid.uuid4())}'
    result.save_crop('input_images', filename)
    files = []
    for file in os.listdir('input_images/badge'):
      # Check if the file name starts with the specified prefix
      if file.startswith(filename):
          # Add the file name to the list
          files.append(file)

    box_count = 0
    for box in result.boxes:
      id = box.id.tolist()[0]
      if id in tracked_badges:
          if tracked_badges[id]['count'] > 10 and not tracked_badges[id]['processing']:
              tracked_badges[id]['processing'] = True

              url = name_url

              payload = {
                  "instances": [
                      {
                          "inputs": {
                              "prompt": "return only the first name on the badge and no other text",
                              "image-url": f'{badge_location}/{files[box_count]}'
                          }
                      }
                  ]
              }

              async with aiohttp.ClientSession() as session:
                  async with session.post(url, json=payload) as response:
                      res = await response.json()
                      if 'Name' in res:
                        tracked_badges[id]['name'] = res['Name']
                      tracked_badges[id]['processing'] = False
                      tracked_badges[id]['count'] = 0
          else:
              tracked_badges[id]['count'] += 1
              tracked_badges[id]['last_detected'] = timestamp
      else:
          tracked_badges[id] = {
              'name': '......',
              'count': 0,
              'last_detected': timestamp,
              'processing': False
            }
      box_count += 1

    for file in os.listdir('input_images/badge'):
      # Check if the file name starts with the specified prefix
      if file.startswith(filename):
          # Construct the full file path
          file_path = os.path.join('input_images/badge', file)
          # Delete the file
          os.remove(file_path)

async def cleanup_old_badges(timestamp):
   for badge in list(tracked_badges.keys()):
      if (timestamp - tracked_badges[badge]['last_detected']) > 10000:
         del tracked_badges[badge]
   await asyncio.sleep(10)

async def main():

  images_path = 'input_images/badge/'
  if os.path.exists(images_path) and os.path.isdir(images_path):
      # Remove the directory and its contents
      shutil.rmtree(images_path)

  args = parse_arguments()
  name_url = args.name_url
  badge_location = args.badge_location
  feed = args.feed
  width, height = args.resolution

  cap = cv2.VideoCapture(feed)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  model = YOLO("weights.pt")

  name = 'unkown'

  # Loop through the video frames
  while cap.isOpened():
      # Read a frame from the video
      success, frame = cap.read()

      if success:
          # Run YOLO inference on the frame
          results = model.track(source=frame, conf=0.4, device='cpu', max_det=1, persist=True)

          #name, prev_name, age = process_result(results[0], vit_model, preprocess, device, name_url, badge_location, prev_name, age)

          if results[0].boxes.id is not None:
            asyncio.create_task(update_tracked_badges(results[0], time.time(), name_url, badge_location))
            asyncio.create_task(cleanup_old_badges(time.time()))

          detections = sv.Detections.from_ultralytics(results[0])

          box_annotator = sv.BoxAnnotator()
          label_annotator = sv.LabelAnnotator()

          labels = []

          for box in results[0].boxes:
             name = '...'
             if box.id and box.id.tolist()[0] in tracked_badges:
                name = tracked_badges[box.id.tolist()[0]]['name']
             labels.append(f'Hello {name} - {round(box.conf.tolist()[0],2)}')

          annotated_image = box_annotator.annotate(scene=frame, detections=detections)
          annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

          # Display the annotated frame
          cv2.imshow("YOLO Inference", annotated_image)

      if (cv2.waitKey(30) == 27):
        break
      await asyncio.sleep(0)

if __name__ == '__main__':
    asyncio.run(main())
