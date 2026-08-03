def load_test_from_npz(self, file, sigma=None, rot=False):
        with np.load(file, allow_pickle=True) as entry:
            src_vs = entry['src_pcd']
            tgt_vs = entry['tgt_pcd']
            # flow = entry['flow']

            src_markers = entry['src_vol']
            tgt_markers = entry['tgt_vol']

            R_gt = entry['R_gt']
            t_gt = entry['t_gt']

            # faces = entry['src_f']
            # edges = entry['src_edges']
            # tgt_f = entry['tgt_f']


            if sigma is not None:
                noise = entry[str(sigma)]
                tgt_vs = tgt_vs + noise

            if rot:
               # rot_src = entry['rot_src']
                rot_tgt = entry['rot_tgt']
                #src_vs = (np.matmul(rot_src, src_vs.T)).T
                tgt_vs = (np.matmul(rot_tgt, tgt_vs.T)).T

                #src_markers = (np.matmul(rot_src, src_markers.T)).T
                tgt_markers = (np.matmul(rot_tgt, tgt_markers.T)).T

                R_gt = np.matmul(R_gt, rot_tgt)
                t_gt = np.matmul(rot_tgt, t_gt)

        return src_vs, tgt_vs, src_markers, tgt_markers, R_gt, t_gt


def pc_normalize(self, pc, centroid=None, m=None):
        if centroid is None:
            centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if m is None:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc, centroid, m

    def ply2np_vox(self, xyz, voxel_size=2, scale=1.0):
        if type(xyz) is str:
            pcd = o3d.io.read_point_cloud(xyz)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_pts = np.asarray(downpcd.points)
        pcd_pts = pcd_pts / scale
        return pcd_pts

    def norm_vox(self, xyz):
        pc, centroid, m = self.pc_normalize(xyz)
        pc_vox = self.ply2np_vox(pc, voxel_size=self.voxel_size, scale=1.0) * m + centroid
        return pc_vox, m

def get_data_np(self, index):
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.load_test_from_npz(self.test_root + self.test_list[index],
                                                                             sigma=self.sigma, rot=self.rot)

        # src_marker = src_raw
        # tgt_marker = src_raw + flow

        src_xyz, m = self.norm_vox(src_raw)
        tgt_xyz, _ = self.norm_vox(tgt_raw)

        s_c = np.mean(src_xyz[:, :3], axis=0)
        t_c = np.mean(tgt_xyz[:, :3], axis=0)

        src_xyz = (src_xyz - s_c) / m
        tgt_xyz = (tgt_xyz - t_c) / m


        src_markers = (src_markers - s_c) / m
        tgt_markers = (tgt_markers - t_c) / m

        return src_xyz, tgt_xyz, src_markers, tgt_markers, s_c, t_c, m, R_gt, t_gt

