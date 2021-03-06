import tensorflow as tf
sess = tf.Session()
from TF_KADP import KADP
from Utils import simple_env
from Utils.ops import compute_return
import numpy as np
import time
import os
'''
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()
'''
np.random.seed(111)
#tf.set_random_seed(111)
print(np.random.rand())
foo = sess.run(tf.random_uniform((1,)))
print('hi',foo)
env = simple_env.Cycle(2)
agent = KADP(env)
check_op = tf.add_check_numerics_ops() 
merged = tf.merge_all_summaries()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/kadp', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train',sess.graph)
sess.run(tf.initialize_all_variables())
cur_time = time.clock()

cumloss = 0
cumgrads = 0
num_steps = int(1e8)
refresh = int(1e2)
target_refresh = int(1e1)
mb_dim = 100
replay_dim = int(mb_dim*1e3)
D_s = np.zeros((replay_dim,agent.s_dim),dtype=np.float32)
D_a = np.zeros((replay_dim,),dtype=np.int32)
D_sPrime = np.zeros((replay_dim,agent.s_dim),dtype=np.float32)
D_aPrime = np.zeros((replay_dim,),dtype=np.int32)
D_rPrime = np.zeros((replay_dim,1),dtype=np.float32)
D_r = np.zeros((replay_dim,1),dtype=np.float32)
D_R = np.zeros((replay_dim,1),dtype=np.float32)
D_nt = np.zeros((replay_dim,1),dtype=np.float32)
D_aPP = np.zeros((replay_dim,),dtype=np.int32)
D_rPP = np.zeros((replay_dim,1),dtype=np.float32)
D_ind = 0
D_full = False
#a = env.action_space.sample()
cur_gamma = 0.0
cur_epsilon = 1.0
def get_mb(mb_s,mb_a,mb_r,mb_sPrime,mb_aPrime,mb_rPrime,mb_nt,mb_R,mb_aPP,mb_rPP):
    mb_s[0,:] = env.reset()
    last_term = -1
    for j in range(mb_dim):
        if j > 1:
            mb_aPrime[j-1]  = mb_a[j]
            mb_rPrime[j-1]  = mb_r[j]
            mb_aPP[j-2]  = mb_a[j]
            mb_rPP[j-2]  = mb_r[j]
            if term:
                mb_s[j,:] = env.reset()
            else:
                mb_s[j,:] = sPrime
        else:
            feed_dict = make_feed_dict()
            cached_V = sess.run(agent.V_view,feed_dict=feed_dict)

        if np.random.rand() < cur_epsilon:
            mb_a[j] = np.random.randint(agent.n_actions)
        else:
            full_r = np.zeros((agent.n_actions,1))
            full_nt = np.zeros((agent.n_actions,1))
            for action in range(agent.n_actions):
                _,full_r[action],term = env.get_transition(mb_s[j],action) 
                full_nt[action] = not term
            feed_dict = make_feed_dict()
            feed_dict.update({agent._full_nt:full_nt,agent._full_r:full_r,agent.V_view:cached_V,agent._s:np.expand_dims(mb_s[j],0)})
            mb_a[j] = sess.run(agent.action,feed_dict=feed_dict)[0]
        sPrime,r,term,_ = env.step(mb_a[j])
        mb_sPrime[j,:] = sPrime
        mb_r[j] = r
        mb_nt[j] = not term
        if term:
            mb_R[last_term+1:j,0] = compute_return(mb_r[last_term+1:j],cur_gamma)
            last_term = j
        if j == (mb_dim-1):
            mb_aPrime[j] = np.random.randint(agent.n_actions)
            _,mb_rPrime[j],term,_ = env.step(mb_aPrime[j])
            mb_aPP[j-1] = mb_aPrime[j]
            mb_rPP[j-1] = mb_rPrime[j]
            mb_aPP[j] = np.random.randint(agent.n_actions)
            _,mb_rPP[j],term,_ = env.step(mb_aPP[j])
    if last_term != (mb_dim-1): #truncate last episode
        mb_R[last_term+1:j,0] = compute_return(mb_r[last_term+1:j],cur_gamma)
    if not agent.change_actions:
        print('mb rewards: ',mb_r.sum(),'pos: ',mb_r[mb_r>0].sum())



max_gamma = .9
gamma_anneal = 0 #int(1e4)
if gamma_anneal > 0:
    gamma = np.linspace(0,max_gamma,gamma_anneal).astype(np.float32)
min_epsilon = 1
epsilon_anneal = -1
if epsilon_anneal > 0:
    epsilon = np.linspace(1,min_epsilon,epsilon_anneal).astype(np.float32)
cumr = 0
cumprob = 0
train = True
def softmax(x,dim=-1):
    ex = np.exp(x)
    denom = np.expand_dims(np.sum(ex,dim),dim)
    return ex/denom
r_hist = []
if 'DISPLAY' in os.environ:
    display = True
    from matplotlib import pyplot as plt
    plt.ion()
else:
    display = False
def plot_stuff():
    mb_latent = env.encode(mb_s)
    plt.figure(1)
    plt.clf()
    if agent.s_dim == 2:
        Xs = mb_latent[:,0]
        Ys = mb_latent[:,1]
        offX = .5*env.radius*np.cos(env.rad_inc*np.arange(agent.n_actions))
        offY = .5*env.radius*np.sin(env.rad_inc*np.arange(agent.n_actions))
        plt.hold(True)
        bub_size = 100
        for action in range(agent.n_actions):
            mask = np.argmax(mb_q_values,0) == action
            #plt.scatter(Xs+offX[action],Ys+offY[action],s=bub_size*mask/2+10)#,c=((mb_q_values[action]-mb_values)))
            plt.scatter(Xs[mask]+offX[action],Ys[mask]+offY[action],s=bub_size/2)
        plt.scatter(Xs,Ys,s=bub_size,c=(mb_values))
        axes = plt.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        plt.hold(False)
        '''database values'''
        fig = plt.figure(2)
        plt.clf()
        mem_latent = env.encode(agent.SPrime_view)
        Xs = mem_latent[:,0]
        Ys = mem_latent[:,1]
        plt.scatter(Xs,Ys,s=100,c=(values))
        axes = fig.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        if agent.z_dim == 2:
            '''model's viewpoint'''
            plt.figure(3)
            plt.clf()
            plt.scatter(mb_embed[:,0],mb_embed[:,1],s=bub_size,c=mb_values)
            plt.figure(4)
            plt.clf()
            plt.scatter(embed[:,0],embed[:,1],s=bub_size,c=values)
    else:
        plt.bar(mb_latent,mb_values)
    plt.figure(6)
    plt.clf()
    plt.plot(r_hist)
    plt.pause(.01)
def log_stuff():
    np.save('r_data',r_hist)
    mem_latent = env.encode(agent.SPrime_view)
    np.save('point_data',mem_latent)
    np.save('val_data',values)
    np.save('viter_data',all_V)
def make_feed_dict():
    return {agent._NTPrime:agent.NTPrime,agent._RPrime:agent.RPrime,agent._RPP:agent.RPP,agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma} 
dqn = False
for i in range(num_steps):
    if i==0 or agent.change_actions:
        get_mb(D_s[D_ind:D_ind+mb_dim],D_a[D_ind:D_ind+mb_dim],D_r[D_ind:D_ind+mb_dim],D_sPrime[D_ind:D_ind+mb_dim],D_aPrime[D_ind:D_ind+mb_dim],D_rPrime[D_ind:D_ind+mb_dim],D_nt[D_ind:D_ind+mb_dim],D_R[D_ind:D_ind+mb_dim],D_aPP[D_ind:D_ind+mb_dim],D_rPP[D_ind:D_ind+mb_dim])
        D_ind = (D_ind + mb_dim) % replay_dim
        if D_ind == 0:
            print('hello!',D_ind)
            D_full = True
    if i < gamma_anneal:
        cur_gamma =gamma[i]
    else:
        cur_gamma = max_gamma
    if i < epsilon_anneal:
        cur_epsilon = epsilon[i]
    else:
        cur_epsilon = min_epsilon
    if i % target_refresh == 0:
        #agent.gen_data(env)
        target_V = sess.run(agent.V_view,feed_dict=make_feed_dict())
    if train:
        if D_full:
            mb_inds = np.random.randint(replay_dim,size=[mb_dim])
        else:
            mb_inds = np.random.randint(D_ind,size=[mb_dim])
        mb_s = D_s[mb_inds]
        mb_a = D_a[mb_inds]
        mb_r = D_r[mb_inds]
        mb_sPrime = D_sPrime[mb_inds]
        mb_aPrime = D_aPrime[mb_inds]
        mb_rPrime = D_rPrime[mb_inds]
        mb_nt = D_nt[mb_inds]
        mb_R = D_R[mb_inds]
        mb_aPP = D_aPP[mb_inds]
        mb_rPP = D_rPP[mb_inds]
        if dqn:
            #information for NEXT state oracle
            full_rPrime = np.zeros((agent.n_actions,mb_dim))
            full_ntPrime = np.zeros((agent.n_actions,mb_dim))
            for a in range(agent.n_actions):
                for s in range(mb_dim):
                    _,full_rPrime[a,s],term = env.get_transition(mb_sPrime[s],a)
                    full_ntPrime[a,s] = not term
            #double DQN -- Q-Prime
            feed_dict = make_feed_dict()
            feed_dict[agent._s] = mb_sPrime
            feed_dict[agent._full_r] = full_rPrime
            feed_dict[agent._full_nt] = full_ntPrime
            full_max_action = sess.run(agent.action,feed_dict=feed_dict)
            feed_dict[agent.V_view] = target_V
            feed_dict[agent._a] = full_max_action
            full_raPrime = np.zeros((mb_dim,1))
            full_ntaPrime = np.zeros((mb_dim,1))
            for s in range(mb_dim):
                full_raPrime[s,0] = full_rPrime[full_max_action[s],s]
                full_ntaPrime[s,0] = full_ntPrime[full_max_action[s],s]
            feed_dict[agent._r] = full_raPrime
            feed_dict[agent._nt] = full_ntaPrime
            target_val = sess.run(agent.q,feed_dict=feed_dict)
            feed_dict = make_feed_dict()
            feed_dict.update({agent._s:mb_s,agent._a:mb_a,agent._sPrime:mb_sPrime,agent._r:mb_r,agent._nt:mb_nt})
            feed_dict[agent.target_val] = target_val
            summary,_,cur_grads,cur_loss,max_prob = sess.run([merged,agent.train_q,agent.get_grads,agent.q_loss,agent.max_prob],feed_dict=feed_dict)
        else:
            feed_dict = make_feed_dict()
            feed_dict.update({agent._s:mb_s,agent._a:mb_a,agent._sPrime:mb_sPrime,agent._r:mb_r,agent._nt:mb_nt})
            feed_dict[agent._aPrime] = mb_aPrime
            feed_dict[agent._rPrime] = mb_rPrime
            feed_dict[agent._aPP] = mb_aPP
            feed_dict[agent._rPP] = mb_rPP
            summary,_,cur_loss,max_prob = sess.run([merged,agent.train_two_step,agent.two_step_loss,agent.max_prob],feed_dict=feed_dict)
            cur_grads = 0.0
        train_writer.add_summary(summary)
        cumprob += max_prob
        cumgrads += cur_grads
        cumloss += cur_loss
    else:
        cumprob += 0
        cumgrads += 0
        cumloss += 0
    if i % refresh == 0:
        '''get mb info'''
        full_r = np.zeros((agent.n_actions,mb_dim))
        full_nt = np.zeros((agent.n_actions,mb_dim))
        for a in range(agent.n_actions):
            for s in range(mb_dim):
                _,full_r[a,s],term = env.get_transition(mb_s[s],a)
                full_nt[a,s] = not term
        feed_dict = make_feed_dict()
        feed_dict.update({agent._s:mb_s,agent._full_r:full_r,agent._full_nt:full_nt})
        all_V = sess.run(agent.all_V,feed_dict=feed_dict)
        mb_q_values,mb_values,mb_actions,values,val_diff,embed,mb_embed,zero_frac \
            = sess.run([agent.q_val,agent.val,agent.action,agent.V_view,
            agent.val_diff,agent.embed(agent.SPrime_view),agent.embed(mb_sPrime),agent.zero_fraction]
            ,feed_dict=feed_dict) 
        '''print stuff'''
        pos_R = agent.R.copy()
        pos_R[pos_R<0] = 0
        print('pos reward stats: ',np.sum(pos_R,1),'net reward stats: ',np.sum(agent.R,1),' mb value stats: ',np.sum(mb_q_values,1),'mb action stats: ',np.histogram(mb_actions,np.arange(agent.n_actions+1))[0])
        steps_per_r = 1/(cumr/mb_dim/refresh+1e-10)
        r_hist.append(cumr)
        print(val_diff,cumprob/refresh,zero_frac,cur_gamma,steps_per_r,'iter: ', i,'loss: ',cumloss/refresh,'grads: ',cumgrads/refresh,'time: ',time.clock()-cur_time)
        if display:
            plot_stuff()
        log_stuff()
        '''reset vars'''
        cumr = 0
        cumprob = 0
        cur_time = time.clock()
        cumloss = 0
        cumgrads = 0
        '''change memories'''
        #need to update target V too!
        #agent.gen_data(env)
    cumr += mb_r.sum()

