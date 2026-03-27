from torch import nn

class CLIPLinearPrecomputed(nn.Module):
    def __init__(
        self,
        list_concepts,
        list_classes,
        device,
    ):
        super().__init__()
        self.name = "CLIP-Linear-Precomputed"
        self.device = device

        # Store concepts and classes
        self.list_concepts = list_concepts
        self.list_classes = list_classes
        
        # Linear parameters
        self.linear = nn.Linear(len(list_concepts), len(list_classes)).to(self.device)

    def forward(self, similarity):
        """
        Forward pass of the model using precomputed features.
        Args:
            indices (torch.Tensor): Indices of the images in the dataset.
        Returns:
            torch.Tensor: Output tensor after forward pass.
        """

        return self.linear(similarity)
