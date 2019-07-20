import torch
import GoogleNet as G

class SimpleBaselineNet(torch.nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, insize, outsize):
        super().__init__()
        # Using the pretrained model for Visual part, here GoogLeNet
        self.GN = G.googlenet(pretrained=True)
        # Outlayer initialized and trained
        self.outlayer = torch.nn.Linear(insize, outsize)

    def forward(self, image, question_encoding):
        imageclass = self.GN(image) # The probabilities of the classes are used, 1000 classes. This has pretty solid information about the image
        if type(imageclass)==tuple:
        	imageclass = imageclass[-1]
        cat = torch.cat([imageclass, question_encoding], dim=1) # Concatenated Image+Question vector to predict Answer
        return self.outlayer(cat)