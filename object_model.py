import os
import trimesh as tm
import plotly.graph_objects as go
import torch
# import pytorch3d.structures
# import pytorch3d.ops
import numpy as np

from torchsdf import index_vertices_by_faces


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda", size=None):

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.surface_points_tensor = None
        
        self.size = size      

    def load_dmax_log(self, log_file_path):
        self.dmax_dict = {}
        with open(log_file_path, "r") as f:
            for line in f:
                code, dmax = line.strip().split(", dmax: ")
                code = code.replace("Code: ", "").replace(".obj", "")
                # print(code)
                self.dmax_dict[code] = float(dmax)
        return self.dmax_dict
    
                
    def initialize(self, object_code_list):

        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        
        for object_code in object_code_list:
            self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False))
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            if self.num_samples != 0:
                # Temporary workaround: use vertices as surface points instead of sampling
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                if len(vertices) > self.num_samples:
                    # Randomly sample vertices
                    indices = torch.randperm(len(vertices))[:self.num_samples]
                    surface_points = vertices[indices]
                else:
                    # Use all vertices and repeat if necessary
                    surface_points = vertices.repeat((self.num_samples + len(vertices) - 1) // len(vertices), 1)[:self.num_samples]
                self.surface_points_tensor.append(surface_points)
        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)
        return None





    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        """
        
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
