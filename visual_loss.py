import numpy as np
import torch
from visdom import Visdom

class Visual_loss(object):
    def __init__(self,win_name='default'):
        self.win_name=win_name
        self.viz=Visdom(env=self.win_name)
        self.initialize_=False
        self.initial_pn=False
        self.initial_pn_val=False
        self.initial_acc_flag=False
        self.initial_acc_flag_val=False
    def initialize(self,total_loss,classify_loss=0,regression_loss=0,X=0):
        self.loss_win=self.viz.line(Y=np.array([total_loss]),X=np.array([X]),opts=dict(title='loss_value'))
        #self.loss_classify_win=self.viz.line(Y=np.array([classify_loss]),X=np.array([X]),opts=dict(title='classify_loss'))
        #self.loss_regress_win=self.viz.line(Y=np.array([regression_loss]),X=np.array([X]),opts=dict(title='regress_loss'))



    def show_loss_curve(self,total_loss,classify_loss=0,regression_loss=0,X=0):

        if not self.initialize_:
            self.initialize_=True
            self.initialize(total_loss,classify_loss,regression_loss,X)
        else:
            self.viz.line(Y=np.array([total_loss]),X=np.array([X]),win=self.loss_win,update='append')
            #self.viz.line(Y=np.array([classify_loss]),X=np.array([X]),win=self.loss_classify_win,update='append')
            #self.viz.line(Y=np.array([regression_loss]),X=np.array([X]),win=self.loss_regress_win,update='append')


    def initial_pos_neg(self,tpr,tnr,epoch):
        self.tpr_win=self.viz.line(X=np.array([epoch]),Y=np.array([tpr]),opts=dict(title='tpr'))
        self.tnr_win=self.viz.line(X=np.array([epoch]),Y=np.array([tnr]),opts=dict(title='tnr'))

    def show_pos_neg_curve(self,tpr,tnr,epoch):
        if not self.initial_pn:
            self.initial_pn=True
            self.initial_pos_neg(tpr,tnr,epoch)
        else:
            self.viz.line(X=np.array([epoch]),Y=np.array([tpr]),win=self.tpr_win,update='append')
            self.viz.line(X=np.array([epoch]),Y=np.array([tnr]),win=self.tnr_win,update='append')
    def initial_pos_neg_val(self,tpr,tnr,epoch):
        self.tpr_win_val=self.viz.line(X=np.array([epoch]),Y=np.array([tpr]),opts=dict(title='tpr_val'))
        self.tnr_win_val=self.viz.line(X=np.array([epoch]),Y=np.array([tnr]),opts=dict(title='tnr_val'))


    def show_pos_neg_curve_val(self,tpr,tnr,epoch):
        if not self.initial_pn_val:
            self.initial_pn_val=True
            self.initial_pos_neg_val(tpr,tnr,epoch)
        else:
            self.viz.line(X=np.array([epoch]),Y=np.array([tpr]),win=self.tpr_win_val,update='append')
            self.viz.line(X=np.array([epoch]),Y=np.array([tnr]),win=self.tnr_win_val,update='append')

    def initial_acc(self,acc,epoch,title_name):
         self.acc_win=self.viz.line(X=np.array([epoch]),Y=np.array([acc]),opts=dict(title=title_name))

    def show_acc(self,acc,epoch,title_name,val=False):
        if not self.initial_acc_flag:
            self.initial_acc_flag=True
            self.initial_acc(acc,epoch,title_name)
        else:
            self.viz.line(X=np.array([epoch]),Y=np.array([acc]),win=self.acc_win,update="append")

    def initial_acc_val(self,acc,epoch,title_name):
         self.acc_win_val=self.viz.line(X=np.array([epoch]),Y=np.array([acc]),opts=dict(title=title_name))


    def show_acc_val(self,acc,epoch,title_name):
        if not self.initial_acc_flag_val:
            self.initial_acc_flag_val=True
            self.initial_acc_val(acc,epoch,title_name)
        else:
            self.viz.line(X=np.array([epoch]),Y=np.array([acc]),win=self.acc_win_val,update="append")


    def show_dice(self,metric,step,title_name):
        self.show_acc(metric,step,title_name)

    def show_dice_val(self,metric,step,title_name):
        self.show_acc_val(metric,step,title_name)
