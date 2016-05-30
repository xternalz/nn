local PenalizedTanh, parent = torch.class('nn.PenalizedTanh','nn.Module')

function TanhShrink:__init(negval)
   parent.__init(self)
   self.negval = negval or (1/4)
   self.tanh = nn.Tanh()
   self.leakyrelu = nn.LeakyReLU(self.negval)
end

function TanhShrink:updateOutput(input)
   self.tanhOutput = self.tanh:updateOutput(input)
   self.output = self.leakyrelu:updateOutput(self.tanhOutput)
   return self.output
end

function TanhShrink:updateGradInput(input, gradOutput)
   local grad = self.leakyrelu:updateGradInput(self.tanhOutput,gradOutput)
   self.gradInput = self.tanh:updateGradInput(input,grad)
   return self.gradInput
end
