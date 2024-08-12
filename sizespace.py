import torch
import math
import torch.nn.functional as F


def getL2distance(x, y):
    return torch.norm(x - y)

# model


class TextSpace(torch.nn.Module):
    def __init__(self, layers_num=1, n_hiper=30):
        super(TextSpace, self).__init__()
        self.n_hiper = n_hiper
        # if len(initial_embedding)<3:
        embeddings = torch.nn.Parameter(torch.randn(
            [layers_num*2 + 1, layers_num*2 + 1, n_hiper+1, 768], requires_grad=True))
        # else:
        #     # got initial embedding
        #     init = torch.load(f"./{initial_embedding}")
        #     init.requires_grad_(True)
        #     embeddings = torch.nn.Parameter(
        #         init.expand([layers_num*2 + 1, layers_num*2 + 1, init.shape[0], init.shape[1]])
        #     )
        self.embeddings = embeddings
        self.layers_num = layers_num

    def load(self, dir):
        self.embeddings = torch.load(dir).requires_grad_(True)

    def interpolate(self, x, y):
        # x ~ [-layers_num, layers_num], y ~ [-layers_num, layers_num]
        # 해당 점에서 왼쪽 위아래, 오른쪽 위아래가 뭔지 알면
        # 어쨌든 floor가 더 작다. -> dx는 항상 양수다. = 왼쪽에서부터 떨어진 거리를 나타냄
        dx = (x - math.floor(x))
        dy = (y - math.floor(y))  # dy는 항상 양수다. = 아래에서부터 떨어진 거리를 나타냄

        x += self.layers_num
        y += self.layers_num

        # 왼쪽 위, 오른쪽 위
        above = self.embeddings[math.floor(x)][math.ceil(
            y)]*(1-dx) + self.embeddings[math.ceil(x)][math.ceil(y)]*(dx)
        # 왼쪽 아래, 오른쪽 아래
        below = self.embeddings[math.floor(x)][math.floor(
            y)]*(1-dx) + self.embeddings[math.ceil(x)][math.floor(y)]*(dx)
        middle = below * (1-dy) + above * (dy)
        return middle

    def check(self):
        check_value = []
        for i in range(3):
            check_value.append(self.embeddings[0][i][-1][:10].clone().detach())
        check_value.append(round(torch.nn.functional.mse_loss(
            self.embeddings[0][0], self.embeddings[0][1], reduction='mean').item(), 10))
        return check_value

    def get_embeddings(self):
        return self.embeddings

    def forward(self, samples, last_point):
        """ 
        sampled from rays, starting_point of ray
        """
        # samples : [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]
        sampled_embeddings = []
        # 현재 sample (x, y) 좌표로 interpolation 해야함.
        for sample in samples:
            x = self.interpolate(sample[0], sample[1])
            sampled_embeddings.append(x)  # 31, 768
        cr = 0
        summed_density = torch.zeros([768]).to("cuda")
        return_embedding = []
        for i, emb in enumerate(sampled_embeddings):
            embedding = emb[:self.n_hiper]  # 30, 768
            density = emb[self.n_hiper]  # 768
            density = F.relu(density)  # 768
            if i < samples.shape[0]-1:
                dists = getL2distance(samples[i], samples[i+1])
            else:
                dists = getL2distance(samples[i], last_point)
            alpha = 1. - torch.exp(-density * dists)
            Ti = torch.exp(-summed_density)
            current_embed = Ti * alpha * embedding
            summed_density += density * dists
            return_embedding.append(current_embed)
        return_embedding = torch.stack(return_embedding)
        return_embedding = torch.sum(return_embedding, dim=0)

        return return_embedding

# model


class Text3DSpace(torch.nn.Module):
    def __init__(self, layers_num=1, n_hiper=30, initial_embedding=""):
        super(Text3DSpace, self).__init__()
        self.n_hiper = n_hiper
        # 5, 5, 5, [30, 768]
        embeddings = torch.nn.Parameter(torch.randn(
            [layers_num*2 + 1, layers_num*2 + 1, layers_num*2 + 1, n_hiper+1, 768], requires_grad=True))
        self.embeddings = embeddings
        self.layers_num = layers_num

    def interpolate(self, x, y, z=0.001):
        # x ~ [-layers_num, layers_num], y ~ [-layers_num, layers_num]
        # 해당 점에서 왼쪽 위아래, 오른쪽 위아래가 뭔지 알면
        # 어쨌든 floor가 더 작다. -> dx는 항상 양수다. = 왼쪽에서부터 떨어진 거리를 나타냄
        dx = (x - math.floor(x))
        dy = (y - math.floor(y))  # dy는 항상 양수다. = 아래에서부터 떨어진 거리를 나타냄
        dz = (z - math.floor(z))  # dz는 항상 양수다. = 아래에서부터 떨어진 거리를 나타냄
        # z_down에서 interpolate, z_up에서 interpolate -> 두개를 합쳐서 또 interpolate

        # print(f"x : {x}, y : {y}, z : {z}, {dx}, {dy}")
        # z_down
        x += self.layers_num
        y += self.layers_num
        z += self.layers_num
        above = self.embeddings[math.floor(x)][math.ceil(y)][math.floor(z)]*(1-dx) + self.embeddings[math.ceil(x)][math.ceil(y)][math.floor(z)]*dx
        below = self.embeddings[math.floor(x)][math.floor(y)][math.floor(z)]*(1-dx) + self.embeddings[math.ceil(x)][math.floor(y)][math.floor(z)]*dx
        middle_low = below*(1-dy) + above*dy

        # z_up
        above = self.embeddings[math.floor(x)][math.ceil(y)][math.ceil(z)]*(1-dx) + self.embeddings[math.ceil(x)][math.ceil(y)][math.ceil(z)]*dx
        below = self.embeddings[math.floor(x)][math.floor(y)][math.ceil(z)]*(1-dx) + self.embeddings[math.ceil(x)][math.floor(y)][math.ceil(z)]*dx
        middle_high = below*(1-dy) + above*dy

        interpolated = middle_low*(1-dz) + middle_high*dz

        return interpolated

    def check(self):
        check_value = []
        for i in range(3):
            check_value.append(self.embeddings[0][i][-1][:10].clone().detach())
        check_value.append(round(torch.nn.functional.mse_loss(
            self.embeddings[0][0], self.embeddings[0][1], reduction='mean').item(), 10))
        return check_value

    def get_embeddings(self):
        return self.embeddings

    def forward(self, samples, last_point):
        """ 
        sampled from rays, starting_point of ray
        """
        # samples : [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]
        sampled_embeddings = []
        # 현재 sample (x, y) 좌표로 interpolation 해야함.
        for sample in samples:
            x = self.interpolate(sample[0], sample[1], sample[2])
            sampled_embeddings.append(x)  # n_hiper+1, 768

        summed_density = torch.zeros([768]).to("cuda")
        return_embedding = []
        for i, emb in enumerate(sampled_embeddings):
            embedding = emb[:self.n_hiper]  # 30, 768
            density = emb[self.n_hiper]  # 768
            density = F.relu(density)  # 768
            if i < samples.shape[0]-1:
                dists = getL2distance(samples[i], samples[i+1])
            else:
                dists = getL2distance(samples[i], last_point)
            alpha = 1. - torch.exp(-density * dists)
            Ti = torch.exp(-summed_density)
            current_embed = Ti * alpha * embedding
            summed_density += density * dists
            return_embedding.append(current_embed)
        return_embedding = torch.stack(return_embedding)
        return_embedding = torch.sum(return_embedding, dim=0)

        return return_embedding


# model
class Text3DSpaceAll(torch.nn.Module):
    def __init__(self, layers_num=1, n_hiper=30, initial_embedding=""):
        super(Text3DSpaceAll, self).__init__()
        self.n_hiper = n_hiper
        embeddings = torch.nn.Parameter(torch.randn(
            [layers_num*2 + 1, layers_num*2 + 1, layers_num*2 + 1, n_hiper+1, 768], requires_grad=True))
        self.embeddings = embeddings
        self.layers_num = layers_num

    def interpolate(self, x, y, z=0.001):
        # x ~ [-layers_num, layers_num], y ~ [-layers_num, layers_num]
        # 범위 몇까지 반영하고 싶냐?
        # 어쨌든 floor가 더 작다. -> dx는 항상 양수다. = 왼쪽에서부터 떨어진 거리를 나타냄
        dx = (x - math.floor(x))
        dy = (y - math.floor(y))  # dy는 항상 양수다. = 아래에서부터 떨어진 거리를 나타냄
        dz = (z - math.floor(z))  # dz는 항상 양수다. = 아래에서부터 떨어진 거리를 나타냄

        x += self.layers_num
        y += self.layers_num
        z += self.layers_num

        def get_line_embedding(floor_x, y, z):
            count=2
            interpolated_x = self.embeddings[floor_x][y][z]*(2-dx) + self.embeddings[floor_x+1][y][z]*(1+dx)
            if floor_x > 0:
                interpolated_x += self.embeddings[floor_x-1][y][z]*(1-dx)
                count+=1
            if floor_x+2 < self.embeddings.shape[0]:
                interpolated_x += self.embeddings[floor_x+2][y][z]*dx
                count+=1
            return interpolated_x/count

        def get_plane_embedding(x, y, z):
            count=2
            above_x = get_line_embedding(math.floor(x), math.ceil(y), z)
            below_x = get_line_embedding(math.floor(x), math.floor(y), z)
            interpolated_plane = below_x * (2-dy) + above_x * (1+dy)

            if math.floor(y) > 0:
                below_two_x = get_line_embedding(math.floor(x), math.floor(y)-1, z)
                interpolated_plane += below_two_x * (1-dy)
                count+=1
            if math.floor(y)+2 < self.embeddings.shape[0]:
                above_two_x = get_line_embedding(math.floor(x), math.ceil(y)+1, z)
                interpolated_plane += above_two_x * dy
                count+=1
            return interpolated_plane/count

        count=2
        interpolated_bottom = get_plane_embedding(x, y, math.floor(z))
        interpolated_top = get_plane_embedding(x, y, math.ceil(z))
        interpolated = interpolated_bottom * (2-dz) + interpolated_top * (1+dz)

        if math.floor(z) > 0:
            interpolated_two_bottom = get_plane_embedding(x, y, math.floor(z)-1)
            interpolated += interpolated_two_bottom * (1-dz)
            count+=1
        if math.floor(z)+2 < self.embeddings.shape[0]:
            interpolated_two_top = get_plane_embedding(x, y, math.ceil(z)+1)
            interpolated += interpolated_two_top * dz
            count+=1

        return interpolated/count

    def check(self):
        check_value = []
        for i in range(3):
            check_value.append(self.embeddings[0][i][-1][:10].clone().detach())
        check_value.append(round(torch.nn.functional.mse_loss(
            self.embeddings[0][0], self.embeddings[0][1], reduction='mean').item(), 10))
        return check_value

    def get_embeddings(self):
        return self.embeddings

    def forward(self, samples, last_point):
        """ 
        sampled from rays, starting_point of ray
        """
        sampled_embeddings = []
        for sample in samples:
            x = self.interpolate(sample[0], sample[1], sample[2])
            sampled_embeddings.append(x)  # n_hiper+1, 768

        summed_density = torch.zeros([768]).to("cuda")
        return_embedding = []
        for i, emb in enumerate(sampled_embeddings):
            embedding = emb[:self.n_hiper]  # 30, 768
            density = emb[self.n_hiper]  # 768
            density = F.relu(density)  # 768
            if i < samples.shape[0]-1:
                dists = getL2distance(samples[i], samples[i+1])
            else:
                dists = getL2distance(samples[i], last_point)
            alpha = 1. - torch.exp(-density * dists)
            Ti = torch.exp(-summed_density)
            current_embed = Ti * alpha * embedding
            summed_density += density * dists
            return_embedding.append(current_embed)
        return_embedding = torch.stack(return_embedding)
        return_embedding = torch.sum(return_embedding, dim=0)

        return return_embedding
