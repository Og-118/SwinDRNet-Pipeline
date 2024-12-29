from inference import SwinDRNetPipeline

model_path = "models/model.pth"
ppl = SwinDRNetPipeline(model_path)
print(ppl.inference([[[1,2,3]]],[[[2]]]))