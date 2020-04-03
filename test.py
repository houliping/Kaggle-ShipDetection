import pandas
import settings
from data import *
import Network.unet_model as unet_model
from torch.backends import cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas
import scipy.misc
config=settings.config
def test_pred(test_path,model_path):
     net=unet_model.get_model(config['channels'],config['classes'])
     cudnn.benchmark = True
     test_dataset=DataSet2(test_path=test_path,phase='test')
     test_dataloader=DataLoader(
         dataset=test_dataset,
         batch_size=1,
         pin_memory=True,
         num_workers=settings.hyper_parameter['num_workers'],
         shuffle=False
     )
     model=torch.load(model_path)
     net.load_state_dict(model['state_dict'])
     net.cuda()
     test_result=[]
     test_name=[]
     net.eval()
     if not os.path.exists("./test_mask/"):
         os.makedirs("./test_mask/")

     for i,(img,img_name) in enumerate(test_dataloader):
         out_pred=net(img.cuda())
         out_pred=(out_pred>0).float()
         out_pred=out_pred.squeeze(0).squeeze(0)
         out_pred=out_pred.contiguous().cpu().detach().numpy()
         scipy.misc.imsave(os.path.join("./test_mask",img_name[0]),out_pred)
         test_name.append(img_name[0])
         test_result.append(encoded(mask=out_pred))
     data={'ImageId':test_name,'EncodedPixels':test_result}
     df=pandas.DataFrame(data)
     df.to_csv('./result_submit.csv')

def encoded(mask:np.ndarray,min_max_threshold=1e-3, max_mean_threshold=None):
    if np.max(mask) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(mask) > max_mean_threshold:
        return '' ## ignore overfilled mask
    mask=mask.T.flatten()
    runs=np.where(mask>0)[0]+1
    ans_start=[]
    ans_length=[]
    length=len(runs)
    i=0
    flag=runs[0]
    while i<length-1:
        if (runs[i+1]-runs[i])==1 :
            i+=1
            if i==length-1:
                encoded_length=runs[i]-flag+1
                ans_length.append(str(encoded_length))
                ans_start.append(flag)
        else:
            encoded_length=runs[i]-flag+1
            ans_length.append(str(encoded_length))
            ans_start.append(str(flag))
            flag=runs[i+1]
            i+=1
            if i==length-1:
                encoded_length=runs[i]-flag+1
                ans_length.append(str(encoded_length))
                ans_start.append(flag)
    ans=''
    for start,length in zip(ans_start,ans_length):
        ans+=' '+str(start)+' '+str(length)
    return ans.lstrip()

if __name__ == '__main__':
    test_pred(config['raw_test_path'],'10.ckpt')