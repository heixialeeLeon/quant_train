import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, dimension):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(input_size= dimension,hidden_size=128,num_layers=2,batch_first=True)
        self.linear1=nn.Linear(in_features=128,out_features=16)
        self.linear2=nn.Linear(16,1)
        self.linear_classify = nn.Linear(16, 2)

    def forward(self,x):
        out,_=self.lstm(x)
        x=out[:,-1,:]
        x=self.linear1(x)
        predict_val=self.linear2(x)
        predict_classify = self.linear_classify(x)
        predict_classify = predict_classify.softmax(1)
        return predict_val, predict_classify

if __name__ == "__main__":
    data = torch.randn(4, 240, 8)
    model = LSTM(8)
    output = model(data)
    print(output[0].shape)
    print(output[1].shape)