local PixelShuffle, parent = torch.class("nn.PixelShuffle", "nn.Module")

-- Periodic shuffling of pixels
-- upscales a [batch x channel*r^2 x m x p] tensor to [batch x channel x r*m x r*p] or
-- downscales a [batch x channel x r*m x r*p] tensor to [batch x channel*r^2 x m x p] tensor
-- tensor, where r is the scaling factor.
-- @param scaleFactor - the scaling factor to use
-- @param upscale - upscale if true else downscale
function PixelShuffle:__init(scaleFactor, upscale)
   parent.__init(self)
   self.scaleFactor = scaleFactor
   self.scaleFactorSquared = self.scaleFactor^2
   print(self.scaleFactor)
   self.upscale = upscale==nil and true or upscale
   local upscaleOrder, downscaleOrder = {1,2,5,3,6,4}, {1,2,4,6,3,5}
   self.permuteOrderFwd = self.upscale and upscaleOrder or downscaleOrder
   self.permuteOrderBwd = self.upscale and downscaleOrder or upscaleOrder
end

-- Computes the forward pass of the layer
-- upscales a [batch x channel*r^2 x m x p] tensor to [batch x channel x r*m x r*p] or
-- downscales a [batch x channel x r*m x r*p] tensor to [batch x channel*r^2 x m x p] tensor
-- @param input - the input tensor to be shuffled
-- @return output - the shuffled tensor
function PixelShuffle:updateOutput(input)
   self._intermediateShape = self._intermediateShape or torch.LongStorage(6)
   self._outShape = self.outShape or torch.LongStorage()
   self._shuffleOut = self._shuffleOut or input.new()

   local batched = false
   local batchSize = 1
   local inputStartIdx = 1
   local outShapeIdx = 1
   if input:nDimension() == 4 then
      batched = true
      batchSize = input:size(1)
      inputStartIdx = 2
      outShapeIdx = 2
      self._outShape:resize(4)
      self._outShape[1] = batchSize
   else
      self._outShape:resize(3)
   end

   local channels = input:size(inputStartIdx)
   local inHeight = input:size(inputStartIdx + 1)
   local inWidth = input:size(inputStartIdx + 2)
   if self.upscale then
      assert(channels % self.scaleFactorSquared == 0, "<PixelShuffle> #channels must be divisible by scaleFactorSquared")
      channels = channels / self.scaleFactorSquared
   else
      assert(inHeight % self.scaleFactor == 0 and
             inWidth % self.scaleFactor == 0, "<PixelShuffle> both height and width must be divisible by scaleFactor")
      inHeight = inHeight / self.scaleFactor
      inWidth = inWidth / self.scaleFactor
   end

   self._intermediateShape[1] = batchSize
   self._intermediateShape[2] = channels
   self._intermediateShape[3] = self.upscale and self.scaleFactor or inHeight
   self._intermediateShape[4] = self.scaleFactor
   self._intermediateShape[5] = self.upscale and inHeight or inWidth
   self._intermediateShape[6] = self.upscale and inWidth or self.scaleFactor

   self._outShape[outShapeIdx] = channels * (self.upscale and 1 or self.scaleFactorSquared)
   self._outShape[outShapeIdx + 1] = inHeight * (self.upscale and self.scaleFactor or 1)
   self._outShape[outShapeIdx + 2] = inWidth * (self.upscale and self.scaleFactor or 1)

   local inputView = torch.view(input, self._intermediateShape)

   self._shuffleOut:resize(inputView:size(self.permuteOrderFwd[1]), inputView:size(self.permuteOrderFwd[2]),
                           inputView:size(self.permuteOrderFwd[3]), inputView:size(self.permuteOrderFwd[4]),
                           inputView:size(self.permuteOrderFwd[5]), inputView:size(self.permuteOrderFwd[6]))
   self._shuffleOut:copy(inputView:permute(table.unpack(self.permuteOrderFwd)))

   self.output = torch.view(self._shuffleOut, self._outShape)

   return self.output
end

-- Computes the backward pass of the layer, given the gradient w.r.t. the output
-- this function computes the gradient w.r.t. the input.
-- @param input - the input tensor
-- @param gradOutput - the tensor with the gradients w.r.t. output
-- @return gradInput - a tensor of the same shape as input, representing the gradient w.r.t. input.
function PixelShuffle:updateGradInput(input, gradOutput)
   self._intermediateShape = self._intermediateShape or torch.LongStorage(6)
   self._shuffleIn = self._shuffleIn or input.new()

   local batchSize = 1
   local inputStartIdx = 1
   if input:nDimension() == 4 then
      batchSize = input:size(1)
      inputStartIdx = 2
   end

   local channels = input:size(inputStartIdx)
   local height = input:size(inputStartIdx + 1)
   local width = input:size(inputStartIdx + 2)
   if self.upscale then
      assert(channels % self.scaleFactorSquared == 0, "<PixelShuffle> #channels must be divisible by scaleFactorSquared")
      channels = channels / self.scaleFactorSquared
   else
      assert(height % self.scaleFactor == 0 and
             width % self.scaleFactor == 0, "<PixelShuffle> both height and width must be divisible by scaleFactor")
      height = height / self.scaleFactor
      width = width / self.scaleFactor
   end

   self._intermediateShape[1] = batchSize
   self._intermediateShape[2] = channels
   self._intermediateShape[3] = self.upscale and height or self.scaleFactor
   self._intermediateShape[4] = self.scaleFactor
   self._intermediateShape[5] = self.upscale and width or height
   self._intermediateShape[6] = self.upscale and self.scaleFactor or width

   local gradOutputView = torch.view(gradOutput, self._intermediateShape)

   self._shuffleIn:resize(gradOutputView:size(self.permuteOrderBwd[1]), gradOutputView:size(self.permuteOrderBwd[2]),
                          gradOutputView:size(self.permuteOrderBwd[3]), gradOutputView:size(self.permuteOrderBwd[4]),
                          gradOutputView:size(self.permuteOrderBwd[5]), gradOutputView:size(self.permuteOrderBwd[6]))
   self._shuffleIn:copy(gradOutputView:permute(table.unpack(self.permuteOrderBwd)))

   self.gradInput = torch.view(self._shuffleIn, input:size())

   return self.gradInput
end


function PixelShuffle:clearState()
   nn.utils.clear(self, {
      "_intermediateShape",
      "_outShape",
      "_shuffleIn",
      "_shuffleOut",
   })
   return parent.clearState(self)
end
