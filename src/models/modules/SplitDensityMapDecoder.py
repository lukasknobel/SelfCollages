from torch import nn


class SplitDensityMapDecoder(nn.Module):

    def __init__(self, map_decoder, count_head, density_scaling):
        super().__init__()
        self.density_scaling = density_scaling
        self.map_decoder = map_decoder
        self.count_head = count_head

    def forward(self, x):

        # call map decoder without cls token
        map = self.map_decoder(x[:, 1:])

        # call count head with cls token
        count = self.count_head(x[:, 0])

        # normalise map
        norm_map = self.density_scaling / map.sum(dim=(-1, -2), keepdim=True)
        map = map * norm_map

        # compute final density map
        density_map = map * count.unsqueeze(-1)

        return density_map
