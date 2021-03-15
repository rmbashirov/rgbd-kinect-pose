import numpy as np
import cv2


from .mymath import align_canonical, rotate_a_b_axis_angle


def get_selected_ids(id_sel_set, req_ids):
    ss_sort = np.argsort(id_sel_set)
    req_sort = np.argsort(req_ids)
    id_ss_srt = id_sel_set[ss_sort]
    id_ss_pos = np.arange(0, len(id_sel_set))[ss_sort]
    req_srt = req_ids[req_sort]
    req_srt_pos = -1 * np.ones(len(req_srt), dtype=int)
    i = 0
    j = 0
    while i < len(id_ss_srt) and j < len(req_srt):
        if req_srt[j] == id_ss_srt[i]:
            req_srt_pos[j] = id_ss_pos[i]
            i += 1
            j += 1
        elif req_srt[j] < id_ss_srt[i]:
            j += 1
        elif id_ss_srt[i] < req_srt[j]:
            i += 1
    req_ids_ans = -1 * np.ones(len(req_srt), dtype=int)
    req_ids_ans[req_sort] = req_srt_pos
    return req_ids_ans


def initialize_pose(joints_kinect_m, joints_viz_f, kinect_smpl, J, parents, dtype, verbose=False):
    #       we take 4 points: pelvis, chest, left shoulder, right shoulder
    #       using these points, we align absolute rotation
    smpl_ids = [0, 12, 16, 17]
    kinect_ids = np.asarray([0, 3, 5, 12], dtype=int)
    kinect_all = kinect_smpl[:, 0]
    #       map smpl joints to kinect joints (only for visible kinect joints)
    s2k = {}
    for i, m in enumerate(kinect_smpl):
        if joints_viz_f[i] == 0:
            continue
        s2k[m[1]] = i

    if verbose:
        print('s2k')
        print(s2k)

    s_kin_ids = get_selected_ids(kinect_all, kinect_ids)
    if np.sum(joints_viz_f[s_kin_ids]) < 4:
        return None, None
    j_s = J[0, smpl_ids]
    j_k = joints_kinect_m[s_kin_ids]
    # first pair defines -y, second pair defines x

    # R_glob, t_glob = orthoprocrustes(j_s, j_k)
    R_smpl = align_canonical(j_s[0] - j_s[1], j_s[3] - j_s[2])
    R_kin = align_canonical(j_k[0] - j_k[1], j_k[3] - j_k[2])

    R_glob = R_kin.T @ R_smpl

    rot, jac = cv2.Rodrigues(R_glob)

    t_glob = -R_glob @ np.mean(j_s, axis=0) + np.mean(j_k, axis=0)
    # t_glob = -R_glob @ j_s[0] + j_k[0]

    n_j = len(parents)
    rots = np.zeros((n_j, 3), dtype=dtype)
    rots_assd = np.zeros(n_j, dtype=np.bool)

    rots[0] = rot.reshape(-1)

    t_glob = t_glob + R_glob @ J[0][0] - J[0][0]

    rots_assd[0] = True
    for ci in range(0, n_j):
        if not ci in s2k:
            continue
        par_i = parents[ci]
        par_chain = [par_i]
        while (not (par_i in s2k)) and (par_i >= 0):
            par_i = parents[par_i]
            par_chain.append(par_i)

        if verbose:
            print(ci, [i.item() for i in par_chain], par_i)

        if (par_i < 0):
            continue
        bone_dir = joints_kinect_m[s2k[ci]] - joints_kinect_m[s2k[par_i]]
        bone_dir /= np.linalg.norm(bone_dir)
        templ_bone_dir = J[0][ci] - J[0][par_i]
        templ_bone_dir /= np.linalg.norm(templ_bone_dir)
        #           we have a child ci in s2k, and a parent par_i there as well
        #           bone_dir is a direction from parent-to child, in the global coordinate system
        #           let's rotate bone_dir to the grandparent's coordinate system
        #           all SMPX rotations are form child to parent
        R_loc2glob = np.eye(3)
        ji = par_i
        while parents[ji] >= 0:
            ji = parents[ji]
            if rots_assd[ji]:
                R, jac = cv2.Rodrigues(rots[ji])
                R_loc2glob = R @ R_loc2glob
        bone_dir_gp = R_loc2glob.T @ bone_dir
        # templ_bone_dir_gp = R_loc2glob.T @ templ_bone_dir
        #           let's find the actual rotation we are going to modify
        rot_id = -1
        R_given = np.eye(3)
        for ni in range(0, len(par_chain)):
            jid = par_chain[len(par_chain) - ni - 1]
            if not rots_assd[jid]:
                rot_id = jid
                break
            else:
                R, jac = cv2.Rodrigues(rots[jid])
                R_given = R_given @ R
        if rot_id < 0:
            continue
        # print('child {} parent {}'.format(ci, par_i))
        # assert (rot_id >= 0)
        bone_dir_gp_act = R_given.T @ bone_dir_gp
        # templ_bone_dir_gp_act = R_given.T @ templ_bone_dir_gp
        aa = rotate_a_b_axis_angle(bone_dir_gp_act, templ_bone_dir)
        # R_init, jac = cv2.Rodrigues(aa)
        # print(np.linalg.norm(R_init @ bone_dir - templ_bone_dir_gp_act))
        rots[rot_id] = -aa
        rots_assd[rot_id] = True
        # return rots, t_glob
    return rots, t_glob


def initialize_pose_advanced(joints_kinect, joints_viz, kinect_verts, J, parents, dtype):
    #bpid, vert_start, vert_end
    bp2kinect = np.asarray(
        [[1, 18, 19], #left hip
         [2, 22, 23],#right hip
         # [3, 0, 1],#spine
         [4, 19, 20],#left knee
         [5, 23, 24],#right knee
         [16, 5, 6],#left shoulder
         [17, 12, 13], #right shoulder
         [18, 6, 7],#left elbow
         [19, 13, 14], #right elbow,
         [12, 3, 26],  # neck
         [15, 26, 27]# head
        ])

    #       we take 4 points: pelvis, chest, left shoulder, right shoulder
    #       using these points, we align absolute rotation
    kinect_ids = np.asarray([0, 3, 5, 12], dtype=int) #, 4, 11, 18, 22

    if np.sum(joints_viz[kinect_ids]) < 4:
        return None, None

    j_k = joints_kinect[kinect_ids]
    j_s = kinect_verts[kinect_ids]
    #first pair defines -y, second pair defines x

    # R_glob, t_glob = orthoprocrustes(j_s, j_k)
    # print(kinect_verts)
    # print(j_s[0] - j_s[1], j_s[3] - j_s[2])
    R_smpl = align_canonical(j_s[0] - j_s[1], j_s[3] - j_s[2])
    R_kin = align_canonical(j_k[0] - j_k[1], j_k[3] - j_k[2])

    R_glob = R_kin.T @ R_smpl

    rot, jac = cv2.Rodrigues(R_glob)
    t_glob = -R_glob @ np.mean(j_s, axis=0) + np.mean(j_k, axis=0)
    # t_glob = -R_glob @ j_s[0] + j_k[0]

    dtype = dtype
    n_j = len(parents)
    rots = np.zeros((n_j, 3), dtype=dtype)
    rots_assd = np.zeros(n_j, dtype=np.bool)

    rots[0] = rot.reshape(-1)
    # print('global rot')
    # print(R_glob)
    t_glob = t_glob + R_glob @ J[0][0] - J[0][0]

    # return rots, t_glob

    for i in range(0, bp2kinect.shape[0]):
        if joints_viz[bp2kinect[i, 1]] + joints_viz[bp2kinect[i, 2]] < 2:
            continue
        model_dir = kinect_verts[bp2kinect[i, 2]] - kinect_verts[bp2kinect[i, 1]]
        model_dir /= np.linalg.norm(model_dir)
        skel_dir = joints_kinect[bp2kinect[i, 2]] - joints_kinect[bp2kinect[i, 1]]
        skel_dir /= np.linalg.norm(skel_dir)
        # print('bpi ' + str(bp2kinect[i, 0]))
        # print('model dir')
        # print(model_dir)
        # print('skel dir')
        # print(skel_dir)
        par_i = parents[bp2kinect[i, 0]]
        rot = np.eye(3)
        while par_i != -1:
            rot_par = rots[par_i]
            rot_par_mat, jac = cv2.Rodrigues(rot_par)
            rot = rot_par_mat @ rot
            par_i = parents[par_i]
        # print('parent rot')
        # print(rot)
        model_dir_in_joint = model_dir
        skel_dir_in_joint = rot.T @ skel_dir
        # print('model in joint coords')
        # print(model_dir_in_joint)
        # print('skel dir in joint')
        # print(skel_dir_in_joint)
        aa = rotate_a_b_axis_angle(skel_dir_in_joint, model_dir_in_joint)
        rots[bp2kinect[i, 0]] = -aa
        rots_assd[bp2kinect[i, 0]] = True

    return rots, t_glob
