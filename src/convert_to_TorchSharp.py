import torch
import utilities.exportsd as exportsd
import ML.agent


sourcefile = "model"
targetfile = sourcefile + ".dat"
sourcefile = sourcefile + ".pt"

model = ML.agent.Model(8)
model.load_state_dict(torch.load(sourcefile))

f = open(targetfile, "wb")
exportsd.save_state_dict(model.state_dict(), f)
f.close()

