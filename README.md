# Deploy the multimodal model
Either run the model directly as a python app or create a container using the dockerfile and run the container.
See the `README.md` in the `multimodal_model` directory.

# Deploy the vision model
See `README.md` in the `vision_model` directory.

# Test the multimodal model

```
curl http://multimodal.example.com:8080/v1/models/internvl2:predict \
-H "Content-Type: application/json" \
-d '{
      "instances": [
      {
        "inputs": {
          "prompt": "return only the first name on the badge and no other text",
          "image-url": "http://vision,example.com:9595/test-image.jpg"
        }
      }]
    }' | jq .
```