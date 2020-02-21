#!/usr/bin/env python
# coding: utf-8

# # load the data

# In[3]:


import scipy.io as sio
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# number = 1
# filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(number) + ".mat")
# imuRaw = io.loadmat(filename)

# load the IMU dataset
imuRaw1 = sio.loadmat("./imu/imuRaw1.mat")
imuRaw2 = sio.loadmat("./imu/imuRaw2.mat")
imuRaw3 = sio.loadmat("./imu/imuRaw3.mat")

# load the Vicon dataset
viconRot1 = sio.loadmat("./vicon/viconRot1.mat")
viconRot2 = sio.loadmat("./vicon/viconRot2.mat")
viconRot3 = sio.loadmat("./vicon/viconRot3.mat")

print("IMU dataset's keys:", imuRaw1.keys())
print("Vicon dataset's keys:", viconRot1.keys())


# # Preprocess the data

# In[11]:


'''
the metric convertor to switch between
Rotation matrix
RPY
quaternion
'''


class UKF:
    def __init__(self):
        pass

    def invQuat(self, quat):
        ## get the input quat
        assert quat.shape == (4,)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]]) / LA.norm(quat)

    def rotVectToQuat(self, rots_vect):
        ## This part need more check,
        ## if the quat in the scipy package is (vect_part, scale_part)
        ## then this part is checked
        # ------------------------------
        ## the output is (4,)
        '''
        angle = LA.norm(rots_vect)
        axis = rots_vect / angle
        # construct the quat
        scale_part = np.cos(angle/2)
        vect_part = axis * np.sin(angle/2)
        q = np.hstack((scale_part,vect_part))
#         print("rotvect to quat:",q)
        assert q.shape == (4,)
        '''
        r = R.from_rotvec(rots_vect)
        q = r.as_quat()
        #         q = q[[3,0,1,2]]
        return q[[3, 0, 1, 2]]

    def quatToRotvect(self, quat):
        q = quat.copy()
        # first change the quat into the function input form
        #         print("quat_to_rotation",quat)
        # read in the quat
        r = R.from_quat(q[[1, 2, 3, 0]])
        # change quat to rotvect
        rot_vect = r.as_rotvec()
        assert rot_vect.shape == (3,)
        return rot_vect

    def SigmaPts(self, P, Q, x):
        ## output should be 12*7
        # dimensionality
        n = P.shape[0]
        d = len(x)
        assert d == 7
        sigma_pts = np.zeros((2 * n, d))
        # add the noise to P
        P_noise = 2 * n * (P + Q)
        # check the PD
        # ---------
        if not np.all(np.linalg.eigvals(P_noise) > 0):
            print("the P_noise is not PSD")
        # ---------
        #         print("P_noise matrix:", P_noise)
        S = LA.cholesky(P_noise).T
        #         print("S matrix:", S)
        assert S.shape == (6, 6)
        for i in range(S.shape[0]):
            # get one col from S, that is (6,)
            col = S[:, i]
            rot_vect = col[:3]
            # change it into the quat form
            #             print("rot vect in SigmaPts, cols:", rot_vect)
            quat = self.rotVectToQuat(rot_vect)
            wi = np.hstack((quat, col[3:]))
            #
            x_pos_q = self.quatMultip(x[:4], wi[:4])
            x_neg_q = self.quatMultip(x[:4], self.invQuat(wi[:4]))
            x_pos_angu = x[4:] + wi[4:]
            x_neg_angu = x[4:] - wi[4:]
            x_pos = np.hstack((x_pos_q, x_pos_angu))
            x_neg = np.hstack((x_neg_q, x_neg_angu))
            #             sigma_pts.append(x_pos)
            #             sigma_pts.append(x_neg)
            sigma_pts[i, :] = x_pos
            sigma_pts[n + i, :] = x_neg
        assert sigma_pts.shape == (12, 7)
        return sigma_pts

    def SigmaPtsMeanCov(self, Sigma_pts, x_pre):
        #         print("Sigma_points in MeanCov function",Sigma_pts)
        ## input Sigma_pts (12*7)
        n_pts, d = Sigma_pts.shape
        # get the previous quat
        quat_pre = x_pre[:4]
        quat_pre = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])
        # max iterations
        max_iter = 100
        # -----------------------------------------
        # start loop
        mean_quat = quat_pre  # init mean
        # interative finding method
        for iters in range(max_iter):
            error_vect = np.zeros((3,))  # init error_vect
            inv_mean_quat = self.invQuat(mean_quat)
            # compute the error vect
            for i in range(n_pts):
                # get the current quat
                quat_i = Sigma_pts[i, :4]

                # compute the error_quat
                #                 inv_mean_quat = self.invQuat(mean_quat)
                error_quat = self.quatMultip(quat_i, inv_mean_quat)
                #                 print("qi",quat_i)
                #                 print("inv_mean_quat", inv_mean_quat)
                #                 print("error_quat", error_quat)

                error_vect = error_vect + self.quatToRotvect(error_quat)
            #                 print("error_vecter",error_vect)
            mean_error_vect = error_vect / n_pts
            mean_error_quat = self.rotVectToQuat(mean_error_vect)
            #             print("mean_error_quat", mean_error_quat)
            # early stopping
            # print("error_vect norm:", LA.norm(mean_error_vect))
            if LA.norm(mean_error_vect) < 1e-8:
                # print("Interitive method find the mean")
                mean_quat = self.quatMultip(mean_error_quat, mean_quat)
                break
            # update the quat
            mean_quat = self.quatMultip(mean_error_quat, mean_quat)
        #             print(mean_quat)

        ## find the mean of angu_vel part
        mean_angu_vel = np.mean(Sigma_pts[:, 4:], axis=0)
        assert mean_angu_vel.shape == (3,)
        # finalize the mean of the sigma points
        mean_sigma_pts = np.hstack((mean_quat, mean_angu_vel))
        # -----------------------------------------

        # compute the covariance matrix of the sigma_pts
        W_matrix = np.zeros((n_pts, 6))
        # change the [X-x_bar] to W
        for i in range(n_pts):
            sigma_pt_i = Sigma_pts[i, :]
            quat_i = sigma_pt_i[:4]
            unbiased_quat_i = self.quatMultip(quat_i, self.invQuat(mean_quat))
            unbiased_rot_vect_i = self.quatToRotvect(unbiased_quat_i)
            unbiased_angu_vel_i = sigma_pt_i[4:] - mean_angu_vel
            # unbias the quat part
            W_matrix[i, :3] = unbiased_rot_vect_i
            W_matrix[i, 3:] = unbiased_angu_vel_i
        covMatrix_sigma_pts = (W_matrix.T @ W_matrix) / n_pts
        # -----------------------------------------

        return mean_sigma_pts, covMatrix_sigma_pts, W_matrix

    def quatMultip(self, q1, q2):
        # this code is checked
        ## this function compute the quaternion multiplication
        # output is (1,4)
        u0 = q1[0]
        v0 = q2[0]
        u = np.array(q1[1:]).reshape(-1, 1)
        v = np.array(q2[1:]).reshape(-1, 1)
        #         print(u)
        #         print(v)
        # q1, q2 is of shape (4,)

        scale_part = u0 * v0 - u.T @ v
        vect_part = u0 * v.T + v0 * u.T + np.cross(u.T, v.T)
        #         print(scale_part)
        #         print(vect_part)
        q = np.hstack((scale_part, vect_part))
        #         print(q)
        q = q.reshape(4, )
        assert q.shape == (4,)
        #         print(q)
        return q

    def processModel(self, sigma_points, delta_t):

        ## this method inputs a state and pass it through the process model in UKF
        # sigma_points (12,7) array
        projected_sigma_points = sigma_points.copy()

        for i in range(sigma_points.shape[0]):
            #             print("state before proj", sigma_points[i,:])
            state = sigma_points[i, :]
            assert state.shape == (7,)
            quat = state[:4]
            angu_vels = state[4:]
            if LA.norm(angu_vels) == 0:
                print("Warning: norm of angu_vels is 0")
            assert quat.shape == (4,)
            # construct delta_quat
            delta_angle = LA.norm(angu_vels) * delta_t
            delta_axis = angu_vels / LA.norm(angu_vels)
            assert delta_axis.shape == (3,)
            delta_quat = np.hstack((np.cos(delta_angle / 2), delta_axis * np.sin(delta_angle / 2)))
            #             print("delta_axisa",delta_axis)
            #             print("sin",np.sin(delta_angle/2))
            assert delta_quat.shape == (4,)
            #             delta_quat = np.hstack((delta_angle,delta_axis))
            # construct the projected quaternion
            #             print("quat_i",quat)
            #             print("delta_quat_i",delta_quat)
            proj_quat = self.quatMultip(quat, delta_quat)
            # construct proj state
            proj_state = np.hstack((proj_quat, angu_vels))
            assert proj_state.shape == (7,)
            #             print("state after change:", state)
            #             print("----------------")
            projected_sigma_points[i, :] = proj_state
        #             print("projected state:", proj_state)
        #             print("state after proj", sigma_points[i,:])

        return projected_sigma_points

    def measureModel(self, proj_sigma_pts, noise_angu_vel, noise_acc):
        ## the output should be N*6
        n_pts, d = proj_sigma_pts.shape
        # gravity
        g = np.array([0, 0, 0, 9.8])
        # init the measure matrix
        mear_matrix = np.zeros((n_pts, 6))
        ## get the acc measurement
        for i in range(n_pts):
            y_i = proj_sigma_pts[i, :]
            quat = y_i[:4]
            # compute the acc measurement
            g_prim_quat_half = self.quatMultip(quat, g)
            g_prim_quat = self.quatMultip(g_prim_quat_half, self.invQuat(quat))
            g_prim = g_prim_quat[1:]
            #             g_prim = self.quatToRotvect(g_prim_quat)
            assert g_prim.shape == (3,)
            acc_mear = g_prim + noise_acc
            # compute the angular velocity measurement
            rot_mear = y_i[4:] + noise_angu_vel
            assert rot_mear.shape == (3,)
            # compose the mear
            mear_i = np.hstack((acc_mear, rot_mear))
            mear_matrix[i, :] = mear_i
        assert mear_matrix.shape == (n_pts, 6)
        return mear_matrix

    def measureMeanCov(self, measure_pts):
        # get the shape from measure
        #         print("mear_matrix", measure_pts)
        n_pts, d = measure_pts.shape
        # get the mean from the measure
        mean_mear = np.mean(measure_pts, axis=0)
        #         print("mean of mearsure matrix",mean_mear)
        unbiased_measure = measure_pts - mean_mear.reshape(1, -1)
        #         print("unbiased mear_matrix", unbiased_measure)
        # compute the covariance matrix
        covMatrix = (unbiased_measure.T @ unbiased_measure) / n_pts

        return mean_mear, covMatrix, unbiased_measure





class convertor:
    def __init__(self):
        pass
    def fromMatrixToRPY(self,rot_matrix):
        ## this method change the rots matrix into RPY
        roll = np.arctan2(rot_matrix[2,1],rot_matrix[2,2])
        pitch = np.arctan2(-rot_matrix[2,0],np.sqrt(rot_matrix[2,1]**2 + rot_matrix[2,2]**2))
        yaw = np.arctan2(rot_matrix[1,0],rot_matrix[0,0])
        return roll, pitch, yaw
    def fromAccToRPY(self,accX, accY, accZ):
        roll = np.arctan2(accY, accZ)
        pitch = np.arctan2(-accX, np.sqrt(accY**2 + accZ**2))
#         roll = np.arctan2(accY, accZ)
#         pitch = np.sin(-accX)
        return roll, pitch
    def fromDeltaTVelToRPY(self,delta_t, wx_vel, wy_vel, wz_vel):
        delta_roll = delta_t * wx_vel
        delta_pitch = delta_t * wy_vel
        delta_yaw = delta_t * wz_vel
        return delta_roll, delta_pitch, delta_yaw


# In[12]:


class RPYVect:
    def __init__(self):
        pass
    def rpytsVectFromVicon(self, Vicon_rots, Vicon_ts):
        # read in the data
        rots_matries = Vicon_rots
        rots_ts = Vicon_ts
        # record the roll-pitch-yaw
        roll_list = []
        pitch_list = []
        yaw_list = []
        # loop through the matries
        # set the convertor
        convt = convertor()
        for i in range(rots_matries.shape[2]):
            matrix = rots_matries[:,:,i]
#             print(matrix)
            # convert the matrix to the rpy system
            roll, pitch, yaw = convt.fromMatrixToRPY(matrix)
            # record rpy
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
        return roll_list, pitch_list, yaw_list, rots_ts
    
    def rpytsVectFromAcc(self,normal_IMU_vals,normal_IMU_ts):
#         print("----------------------------------------------------")
#         print(normal_IMU_vals[:,:3])
        # get the acc part of the normal_IMU_dict
        accs = normal_IMU_vals[:,:3]
#         print("accs shape",accs.shape)
        # record the roll & pitch
        roll_list = []
        pitch_list = []
        # set the convertor
        convt = convertor()
        for i in range(accs.shape[0]):
            accX, accY, accZ = accs[i,:]
            # compute the roll & pitch
            roll, pitch = convt.fromAccToRPY(accX, accY, accZ)
            roll_list.append(roll)
            pitch_list.append(pitch)
        return roll_list, pitch_list, normal_IMU_ts
    
    
    def rpytsVectFromAnguVel(self,init_rpy,normal_IMU_vals,normal_IMU_ts):
        # record the init rpy
        init_roll, init_pitch, init_yaw = init_rpy
        roll_list = [init_roll]
        pitch_list = [init_pitch]
        yaw_list = [init_yaw]
        
        # get the angu_vel part of the normal_IMU_dict
        angu_Vels = normal_IMU_vals[:,3:]
        angu_Vels_X = normal_IMU_vals[:,3]
        angu_Vels_Y = normal_IMU_vals[:,4]
        angu_Vels_Z = normal_IMU_vals[:,5]
        
        ## loop over the time and integrate the angular velocity
        # set the convertor
        convt = convertor()
        for i in range(1, angu_Vels.shape[0]):
            # get the delta time
            delta_t = normal_IMU_ts[i] - normal_IMU_ts[i-1]
            # get the angular velocity
#             print(angu_Vels_X.shape)
            angu_velX = angu_Vels_X[i-1]
            angu_velY = angu_Vels_Y[i-1]
            angu_velZ = angu_Vels_Z[i-1]
#             print("angular velocity X:",angu_velX)
#             print("angular velocity Y:",angu_velY)
#             print("angular velocity Z:",angu_velZ)
            delta_roll, delta_pitch, delta_yaw = convt.fromDeltaTVelToRPY(delta_t, angu_velX, angu_velY, angu_velZ)
            
            # add the delta to the list
            roll_list.append(delta_roll)
            pitch_list.append(delta_pitch)
            yaw_list.append(delta_yaw)
        # cumsum the list to get the angulars at each time stamp    
        roll_list = np.cumsum(roll_list)
        pitch_list = np.cumsum(pitch_list)
        yaw_list = np.cumsum(yaw_list)
        return roll_list, pitch_list, yaw_list, normal_IMU_ts
            
        # 
        


# In[259]:


def estimate_rot(data_num=1):
    from scipy.spatial.transform import Rotation as R
    #your code goes here
    ### data loader
    #---------------------------------------------------------------------
    # load the dataset
    number = data_num
    imuRaw = sio.loadmat("./imu/imuRaw" + str(number) + ".mat")
    viconRot = sio.loadmat("./vicon/viconRot" + str(number) + ".mat")
    # load the imu vals & ts
    dataset_vals = imuRaw['vals'].copy()
    dataset_ts = imuRaw['ts'].copy()
    # load the vicon rots & ts
    Vicon_rots = viconRot['rots'].copy()
    Vicon_ts = viconRot['ts'].copy()
    # data samples count
    _, N = dataset_vals.shape
#     print(type(dataset_vals))
    
    ### Pre-process
    #---------------------------------------------------------------------
    ## transform the first 3 dimentions of the dataset, get 3*N dataset
    # define the bias & sensitivity factor for the acc part
    bias = np.array([[np.mean(dataset_vals[0,:100])],
                     [np.mean(dataset_vals[1,:100])],
                     [np.mean(dataset_vals[2,:100])-(np.mean(dataset_vals[2,:100])-np.mean(dataset_vals[1,:100]))]])
#     print("x bias",np.mean(dataset_vals[0,:100]))
#     print("y bias",np.mean(dataset_vals[1,:100]))
#     print("z bias",np.mean(dataset_vals[2,:100]))
#     print(bias.shape)
    
    sens_factor = 0.08
    scale_factor = 3300 / (1023*sens_factor)
    scale_factor = 0.095
    # transform the first 3 dimentions of the dataset, get 3*N dataset
#     print((dataset_vals[:3,:] - bias)*scale_factor)
    imu_acc = (dataset_vals[:3,:] - bias)*scale_factor
    imu_acc[:2,:] = -imu_acc[:2,:]
    print('IMU ACC data:\n', imu_acc)
    '''
    plt.plot(imu_acc[0,:],label="x")
    plt.plot(imu_acc[1,:],label="y")
    plt.plot(imu_acc[2,:],label="z")
    plt.legend()
    plt.show()
    '''
    print("acceleration check:\n", np.sqrt(imu_acc[0,:]**2 + imu_acc[1,:]**2 + imu_acc[2,:]**2 ))
    '''
    plt.plot(np.sqrt(imu_acc[0,:]**2 + imu_acc[1,:]**2 + imu_acc[2,:]**2 ))
    plt.show()
    '''
    
    ## transform the latter 3 dimention of the dataset, get 3*N dataset
    # define the bias & sensitivity factor for the acc part
    bias = np.array([[np.mean(dataset_vals[3,:100])],
                     [np.mean(dataset_vals[4,:100])],
                     [np.mean(dataset_vals[5,:100])]])
    sens_factor = 1
    scale_factor = 3.3/(1023*sens_factor)
#     print((dataset_vals[3:,:] - bias)*scale_factor)
    imu_rot = (dataset_vals[3:,:] - bias)*0.015
    
#     print(imu_rot)
    imu_rot[[0,1,2]] = imu_rot[[1,2,0]]
    print("imu angu",imu_rot)
#     print("norm2 check:", np.sqrt(imu_rot[0,:]**2 + imu_rot[1,:]**2 + imu_rot[2,:]**2 ))
    '''    
    plt.plot(imu_rot[0,:])
    plt.title("imu_rot1")
    plt.show()
    plt.plot(imu_rot[1,:])
    plt.title("imu_rot2")
    plt.show()
    plt.plot(imu_rot[2,:])
    plt.title("imu_rot0")
    plt.show()
    '''
    # concatenate these two transformed dataset
    print("concatenate these two transformed dataset")
    dataset = np.concatenate((imu_acc, imu_rot), axis=0).T
    assert dataset.shape == (N,6)
#     dataset[:,:2] = - dataset[:,:2]
    normal_IMU_vals = dataset
    print("normal_IMU_vals shape:", normal_IMU_vals.shape)
    print(normal_IMU_vals[0,3],normal_IMU_vals[0,4],normal_IMU_vals[0,5])
    normal_IMU_ts = dataset_ts[0]
#     print(normal_IMU_vals)

#     print(dataset_ts)
#     plt.plot(dataset_ts[0])
#     plt.show()
    
    
    ### plot RPY from IMU_acc, IMU_w, and Vicon_rots
    
    RPYVect_constructor = RPYVect()
    Vicon_roll_list, Vicon_pitch_list, Vicon_yaw_list, rots_ts = RPYVect_constructor.rpytsVectFromVicon(Vicon_rots,Vicon_ts)

#     plt.show()
    Acc_roll_list, Acc_pitch_list, normal_IMU_ts = RPYVect_constructor.rpytsVectFromAcc(normal_IMU_vals,normal_IMU_ts)
    
    Angu_roll_list, Angu_pitch_list, Angu_yaw_list, normal_IMU_ts = RPYVect_constructor.rpytsVectFromAnguVel([Vicon_roll_list[0], Vicon_pitch_list[0], Vicon_yaw_list[0]],normal_IMU_vals,normal_IMU_ts)
    
    '''
    plt.plot(Angu_roll_list,label="roll_Angu")
    plt.plot(Vicon_roll_list,label="roll_Vicon")
    plt.plot(Acc_roll_list,label="roll_Acc")
    plt.title("Roll Plot")
    plt.xlabel("time")
    plt.legend()
    plt.show()
    
    plt.plot(Angu_pitch_list,label="pitch_Angu")
    plt.plot(Vicon_pitch_list,label="pitch_Vicon")
    plt.plot(Acc_pitch_list,label="pitch_Acc")
    plt.title("Pitch Plot")
    plt.xlabel("time")
    plt.legend()
    plt.show()
    
    plt.plot(Angu_yaw_list,label="yaw_Angu")
    plt.plot(Vicon_yaw_list,label="yaw_Vicon")
    plt.title("Yaw Plot")
    plt.xlabel("time")
    plt.legend()
    plt.show()
    '''
    
    print("time lens:",normal_IMU_ts.shape)
    
    ### Start UKF
    #---------------------------------------------------------------------
    # init the state, P
    x = np.array([1,0,0,0,normal_IMU_vals[0,3],normal_IMU_vals[0,4],normal_IMU_vals[0,5]])
    # use a random matrix A to generate a PD matrix
    seed = np.random.seed(1)
#     A = np.random.rand(6,6)
#     P = A@A.T
#     P = 1*np.eye(6)

    P = np.array([[1,0,0,0,0,0],
                  [0,1,0,0,0,0],
                  [0,0,1,0,0,0],
                  [0,0,0,1e-6,0,0],
                  [0,0,0,0,1e-6,0],
                  [0,0,0,0,0,1e-6]])
    # set a noise PD matrix Q
    Q_noise = 1e-10
    # Q = Q_noise*np.eye(6)
    Q = np.array([[Q_noise,0,0,0,0,0],
                  [0,Q_noise,0,0,0,0],
                  [0,0,Q_noise,0,0,0],
                  [0,0,0,Q_noise,0,0],
                  [0,0,0,0,Q_noise,0],
                  [0,0,0,0,0,Q_noise]])

    R_noise = np.array([[1e-3,0,0,0,0,0],
                        [0,1e-3,0,0,0,0],
                        [0,0,1e-3,0,0,0],
                        [0,0,0,1e-5,0,0],
                        [0,0,0,0,1e-5,0],
                        [0,0,0,0,0,1e-5]])

    noise_acc = 0*np.random.normal(0, 1, 3)
    noise_angu_vel = 0*np.random.normal(0, 1, 3)
    # start the ukf filter
    ukf = UKF()
    pred_r = []
    pred_p = []
    pred_y = []
    from scipy.spatial.transform import Rotation as R
    for i in range(1, normal_IMU_ts.shape[0]):
        if i % 1000 == 999:
            print("UKF cycles", i+1)
        # compute the delta_t for process model use
        delta_t = normal_IMU_ts[i] - normal_IMU_ts[i-1]
        # get the sigma points
        sigma_pts = ukf.SigmaPts(P, Q, x)
        # -----------------------------------
        # ## test for init mear
        # init_mear = ukf.measureModel(sigma_pts, noise_angu_vel, noise_acc)
        # print("init sigma mear",init_mear)
        # init_z,_,_ = ukf.measureMeanCov(init_mear)
        # print("init mear", init_z)
        # -----------------------------------
        n_pts, _ = sigma_pts.shape
        # project the sigma_pts
        proj_sigma_pts = ukf.processModel(sigma_pts, delta_t)
        # get the mean cov from the projected
        x_hat_bar, P_bar, W_prime = ukf.SigmaPtsMeanCov(proj_sigma_pts,x)
        
        
        mear_pts = ukf.measureModel(proj_sigma_pts, noise_angu_vel, noise_acc)
        true_mear = normal_IMU_vals[i,:]
        print("true mear",true_mear)

        assert true_mear.shape == (6,)
        
        z_bar, P_zz, Z = ukf.measureMeanCov(mear_pts)
        print("pred mear", z_bar)
#         P_xz = Z.T @ W_prime / n_pts
        P_xz = (W_prime.T @ Z) / n_pts
        P_vv = P_zz + R_noise
        K = P_xz @ LA.pinv(P_vv)
        ## Updated x & P
        # update x
        state_cali_6d = K @ (true_mear-z_bar)
        assert state_cali_6d.shape == (6,)
        state_cali_7d = np.hstack((ukf.rotVectToQuat(state_cali_6d[:3]), state_cali_6d[3:]))
        assert state_cali_7d.shape == (7,)
#         print(state_cali_6d.shape)
        x_pred_quat = ukf.quatMultip(x_hat_bar[:4], state_cali_7d[:4])
        x_pred_angu = x_hat_bar[4:] + state_cali_7d[4:]
        x = np.hstack((x_pred_quat,x_pred_angu))
        # update P
        P = P_bar - K @ P_vv @ K.T
        
        
#         print("eigen-value of ",np.linalg.eigvals(P_xz))
#         if not np.all(np.linalg.eigvals(LA.pinv(P_xz)) > 0):
#             print("the P_xz is not PSD")
#         #---------
        
#         #---------
#         if not np.all(np.linalg.eigvals(P) > 0):
#             print("the P is not PSD")
#         #---------
        
        # quat part of x
        x_q = x[:4]
#         print(x_q)
        r = R.from_quat(x_q[[1,2,3,0]])
        r,p,y = r.as_euler('xyz', degrees=False)
        pred_r.append(-r)
        pred_p.append(-p)
        pred_y.append(y)
        
        see = 5000
        if i == see:
            break
#         print(x.shape)
#         print(P.shape)
    plt.plot(pred_r, label="pred_r")
    plt.plot(Vicon_roll_list[:see], label="vicon_r")
    plt.title("Roll compara")
    plt.legend()
    plt.show()
    plt.plot(pred_p, label="pred_p")
    plt.plot(Vicon_pitch_list[:see], label="vicon_p")
    plt.title("Pitch compara")
    plt.legend()
    plt.show()
    plt.plot(pred_y, label="pred_y")
    plt.plot(Vicon_yaw_list[:see], label="vicon_y")
    plt.title("Yaw compara")
    plt.legend()
    plt.show()
        
#         if i == 1:
#             break
        
        
        # compute the delta_t for process model use
#         delta_t = normal_IMU_ts[i] - normal_IMU_ts[i-1]
#         print("UKF iteration", i)
        # get the sigma points
#         sigma_pts = ukf.SigmaPts(P, Q, x)
        # project the sigma points
#         proj_sigma_pts = ukf.processModel(sigma_pts,delta_t)
        # get the 
        
    
    
    
    
#     # init the state, P
# # x = np.array([1,0,0,0,0.0056,0.012,0.0051])
# x = np.array([1,0,0,0,0.01,0.01,0.01])
# # use a random matrix A to generate a PD matrix
# seed = np.random.seed(1)
# # test Sigma_pts
# A = np.random.rand(6,6)
# P = np.eye(6)
# # set a noise PD matrix Q
# Q = np.eye(6)
# R = 0.01*np.eye(6)
# noise_angu_vel = np.random.normal(0,1,3)
# noise_acc = np.random.normal(0,1,3)
# sigma_pts = test.SigmaPts(P, Q, x)
# n_pts, d_7 = sigma_pts.shape
# print("sigma_pts:\n",sigma_pts)
# # print("sigma_pts shape:", sigma_pts.shape)
# proj_sigma_pts = test.processModel(sigma_pts,0.1)
# print("sigma_pts_after_proj(should no change):\n",sigma_pts)
# x_hat_bar, P_k_bar, W_prime = test.SigmaPtsMeanCov(proj_sigma_pts,x)
# mear_pts = measureModel(proj_sigma_pts, noise_angu_vel, noise_acc)
# z_bar, P_zz, Z = test.measureMeanCov(proj_sigma_pts)
# P_xz = Z.T @ W_prime / n_pts
# P_vv = P_zz + R
# K = P_xz @ LA.pinv(P_vv)
# x = x_hat_bar + K@()
    
    
    pass

#     return roll,pitch,yaw
estimate_rot(1)


# # Process model

# In[254]:

# number = 1
# imuRaw = sio.loadmat("./imu/imuRaw" + str(number) + ".mat")
# viconRot = sio.loadmat("./vicon/viconRot" + str(number) + ".mat")   
    
# imu_ts = imuRaw["ts"]
# imu_vals = imuRaw["vals"]
    
# test = UKF()
# # print("quatMulti checked")
# print("test for multiplication:",test.quatMultip([1,0,0,0],[0,0,0,9.8]))
# half = test.quatMultip([1,0,0,0],[0,0,0,9.8])
# whole = test.quatMultip(half,[1,0,0,0])
# print(whole)
# r = R.from_quat(whole)
# print(r.as_rotvec())
# test_quat = test.rotVectToQuat([np.pi/3,np.pi/3,0])
# r = R.from_rotvec([np.pi/3,np.pi/3,0])
# print("scipy",r.as_quat())
# print("scipy",r.as_quat()[[3,0,1,2]])
# print("scipy",r.as_rotvec().shape)
# print("test quat:", test_quat)
# inv_test_quat = test.invQuat(test_quat)
# print("inv test quat:", inv_test_quat)
# # print(np.cross([2,3,4],[3,2,4]))

# # init the state, P
# # x = np.array([1,0,0,0,0.0056,0.012,0.0051])
# x = np.array([1,0,0,0,0.01,0.01,0.01])
# # use a random matrix A to generate a PD matrix
# seed = np.random.seed(1)
# # test Sigma_pts
# A = np.random.rand(6,6)
# P = np.eye(6)
# # set a noise PD matrix Q
# Q = np.eye(6)
# R_noise = 0.01*np.eye(6)
# noise_angu_vel = np.random.normal(0,1,3)
# noise_acc = np.random.normal(0,1,3)
# sigma_pts = test.SigmaPts(P, Q, x)
# n_pts, d_7 = sigma_pts.shape
# print("sigma_pts:\n",sigma_pts)
# # print("sigma_pts shape:", sigma_pts.shape)
# proj_sigma_pts = test.processModel(sigma_pts,0.1)
# print("sigma_pts_after_proj(should no change):\n",sigma_pts)
# x_hat_bar, P_k_bar, W_prime = test.SigmaPtsMeanCov(proj_sigma_pts,x)
# mear_pts = measureModel(proj_sigma_pts, noise_angu_vel, noise_acc)
# z_bar, P_zz, Z = test.measureMeanCov(proj_sigma_pts)
# P_xz = Z.T @ W_prime / n_pts
# P_vv = P_zz + R_noise
# K = P_xz @ LA.pinv(P_vv)
# x = x_hat_bar + K@()



