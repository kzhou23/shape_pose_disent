import numpy as np
import os
from psbody.mesh import Mesh

def output_meshes(meshes, epoch, config, samples=5):
    data = meshes[:samples].reshape(samples, -1, 6890, 3)

    bm_path = config['path']['body_models']
    data_path = config['path']['processed_data']
    vis_path = config['path']['visualization']

    bm = np.load(os.path.join(bm_path, 'male/model.npz'))

    if not os.path.isdir(os.path.join(vis_path, 'epoch_{}'.format(epoch))):
        os.mkdir(os.path.join(vis_path, 'epoch_{}'.format(epoch)))

    for i in range(samples):
        for j in range(data.shape[1]):
            mesh = Mesh(v=data[i, j], f=bm['f'])
            mesh.write_ply(os.path.join(vis_path, 'epoch_{}/{}_{}.ply'.format(epoch, i, j)))


    