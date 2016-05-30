local Padding, parent = torch.class('nn.Padding', 'nn.Module')

-- pad puts in [pad] amount of [value] over dimension [dim], starting at index [index] in that dimension. If pad<0, index counts from the left.  If pad>0 index counts from the right
-- index = 1 pads before index 1.  index = 2 pads starting before index 2 and after index 1 in dimension [dim]
-- setting step size
function Padding:__init(dim, pad, nInputDim, value, index, step)
   self.value = value or 0
   self.index = index or 1
   self.step = math.abs(step) or 0
   self.dim = dim
   self.pad = pad
   self.nInputDim = nInputDim
   self.outputSize = torch.LongStorage()
   parent.__init(self)
end

function Padding:updateOutput(input)
   self.outputSize:resize(input:dim())
   self.outputSize:copy(input:size())
   local dim = self.dim
   if self.nInputDim and input:dim() ~= self.nInputDim then
      dim = dim + 1
   end
   self.outputSize[dim] = self.outputSize[dim] + math.abs(self.pad)
   local nRepeat = 1
   local remainder = 0
   if self.step ~= 0 then
      if self.step > 0 then
         nRepeat = index
      else
         nRepeat = input:dim()-index+1
      end
      nRepeat = math.floor(nRepeat/math.abs(self.step))
      remainder = nRepeat%math.abs(self.step)
      if nRepeat > 1 then
         self.outputSize[dim] = self.outputSize[dim] + (nRepeat-1)*math.abs(self.pad)
      end
   end
   self.output:resize(self.outputSize)
   self.output:fill(self.value)
   local startIndex
   if step <= 0 and index ~= 1 then
      startIndex = 1
   elseif step <= 0 and index == 1 then
      startIndex = pad+1
   elseif step > 0 and index ~= 1 then
      startIndex = output:size(dim)
   elseif step <= 0 and index == 1 then
      startIndex = output:size(dim)-pad
   end
   for i=1,nRepeat+1 do
   end
   local pad = self.pad
   local step = self.step
   if index ~= 1 and step >= 0 then
      self.output:narrow(dim, 1, index-1):copy(input:narrow(dim, 1, index - 1))
   elseif index ~= input:size(dim) and step < 0 then
      self.output:narrow(dim, index + pad, input:size(dim) - (index - 1)):copy(input:narrow(dim, index, input:size(dim) - (index - 1)))
   end
   for i=1,nRepeat do
      if index == 1 then
         self.output:narrow(dim, 1 + pad, length):copy(input)
      elseif index == input:size(dim) + 1 then
         self.output:narrow(dim, 1, length):copy(input)
      else
         self.output:narrow(dim, 1, index - 1):copy(input:narrow(dim, 1, index - 1))
         self.output:narrow(dim, index + pad, input:size(dim) - (index - 1)):copy(input:narrow(dim, index, input:size(dim) - (index - 1)))
      end
      index = index + step
   end
   return self.output
end

function Padding:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   local dim = self.dim
   if self.nInputDim and input:dim() ~= self.nInputDim then
      dim = dim + 1
   end
   local index = self.index
   local pad = self.pad
   if pad > 0 then
      index = input:size(dim) - index + 2
   else
      pad = -pad
   end
   if index == 1 then
      self.gradInput:copy(gradOutput:narrow(dim, 1 + pad, input:size(dim)))
   elseif index == input:size(dim) + 1 then
      self.gradInput:copy(gradOutput:narrow(dim, 1, input:size(dim)))
   else
      self.gradInput:narrow(dim, 1, index - 1):copy(gradOutput:narrow(dim, 1, index - 1))
      self.gradInput:narrow(dim, index, input:size(dim) - (index - 1)):copy(gradOutput:narrow(dim, index + pad, input:size(dim) - (index - 1)))
   end
   return self.gradInput
end
