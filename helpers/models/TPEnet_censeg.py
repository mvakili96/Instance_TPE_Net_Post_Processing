# 2020/7/21
# Jungwon Kang

# Note that the term 'TPEnet' and 'rpnet' are mixed, but both terms are used to indicate 'TPEnet'.


import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################################################################
class ConvLayer(nn.Sequential):
    """ConvLayer - CBR module, where CBR: Conv + BN + ReLU"""

    ###======================================================================================================
    ### ConvLayer::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
    #end


    ###======================================================================================================
    ### ConvLayer::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end


########################################################################################################################
class HarDBlock(nn.Module):
    """HarDBlock"""

    ###======================================================================================================
    ### HarDBlock::get_link()
    ###======================================================================================================
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        #end

        out_channels = growth_rate

        link = []

        for i in range(10):
            dv = 2 ** i

            if layer % dv == 0:
                k = layer - dv
                link.append(k)

                if i > 0:
                    out_channels *= grmul
                #end
            #end
        #end

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        #end

        return out_channels, in_channels, link
    #end


    ###======================================================================================================
    ### HarDBlock::get_out_ch()
    ###======================================================================================================
    def get_out_ch(self):
        return self.out_channels
    #end


    ###======================================================================================================
    ### HarDBlock::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()

        ###
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0       # if upsample else in_channels


        ###
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out

            layers_.append( ConvLayer(inch, outch) )

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
            #end


            # print("idx_layers: [%d]" % i)
            # print(link)
            # print("inch: [%d], outch: [%d]" % (inch, outch))
        #end


        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
    #end


    ###======================================================================================================
    ### HarDBlock::forward()
    ###======================================================================================================
    def forward(self, x):

        layers_ = [x]

        for layer in range(len(self.layers)):
            ###
            link = self.links[layer]
            tin = []


            ###
            for i in link:
                tin.append(layers_[i])
            #end


            ###
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            #end


            ###
            out = self.layers[layer](x)
            layers_.append(out)
        #end

        t = len(layers_)

        out_ = []

        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
            #end
        #end

        out = torch.cat(out_, 1)
        return out
    #end


########################################################################################################################
class TransitionUp(nn.Module):
    """TransitionUp"""

    ###======================================================================================================
    ### TransitionUp::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)
    #end

    ###======================================================================================================
    ### TransitionUp::forward()
    ###======================================================================================================
    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)

        if concat:                            
            out = torch.cat([out, skip], 1)
        #end
          
        return out
    #end
#end


########################################################################################################################
class TPEnet_censeg(nn.Module):
    """TPEnet"""

    ############################################################################################################
    ### TPEnet_a::__init__()
    ############################################################################################################
    def __init__(self, n_classes=3):
        super(TPEnet_censeg, self).__init__()

        ###================================================================================================
        ### parameters (for FC-HarDNet)
        ###================================================================================================

        ### output ch of init-conv
        first_ch = [16, 24, 32, 48]

        ### output of [HDB-xD & TDx], where HDB-xD is HDB-xDown, TDx is Transition Down-x
        ch_list  = [64, 96, 160, 224, 320]

        ###
        grmul    = 1.7

        ### output of first bottleneck layer in each [HDB-xD & TDx]
        gr       = [10, 16, 18, 24, 32]

        ### the number of bottleneck layers in each [HDB-xD & TDx]
        n_layers = [4, 4, 8, 8, 8]

        ###
        blks = len(n_layers)            # => blks: 5


        ###================================================================================================
        ### parameters (for rpnet)
        ###================================================================================================
        ch_output_fc_hardnet_a = 48         # output of HDB-3U
        ch_output_fc_hardnet_b = 3         # output of Conv-Final


        ###================================================================================================
        ### init modules (for rpnet)
        ###================================================================================================
        #self.rpnet_decoder_cnvs = None
        self.rpnet_decoder_hmap_center = None


        ###================================================================================================
        ### init modules (for FC-HarDNet)
        ###================================================================================================
        self.base = nn.ModuleList([])


        ###================================================================================================
        ### set init-conv (for FC-HarDNet)
        ###================================================================================================
        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2) )  # conv0
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )                          # conv1
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )                # conv2
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )                          # conv3
            # created
            #   (conv0): Conv2d( 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv1): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            #   (conv2): Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv3): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


        ###================================================================================================
        ### down-network
        ###================================================================================================
        self.shortcut_layers           = []     # self.shortcut_layers          = [4, 7, 10, 13]
        self.set_idx_module_HarDBlock  = []     # self.set_idx_module_HarDBlock = [4, 7, 10, 13, 16]
        skip_connection_channel_counts = []

        ###
        ch = first_ch[3]                        # output ch of last conv in init-conv

        ###
        for i_blk in range(blks):
            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<FC-HarDNet - base>>
            ###///////////////////////////////////////////////////////////////////////////////////////

            ###------------------------------------------------------------------------------
            ### create HarDBlock
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], ch_in:[%d]' % (i,ch))
            blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk])
            ch = blk.get_out_ch()
            ch_out_HarDBlock_this = blk.get_out_ch()
            #print('idx_blk:[%d], ch_out:[%d]' % (i,ch))


            ###------------------------------------------------------------------------------
            ### append HarDBlock & create shortcut
            ###------------------------------------------------------------------------------

            ### append ch into skip_connection_channel_counts
            skip_connection_channel_counts.append(ch)

            ### append HarDBlock
            self.base.append( blk )                                               # APPEND

            ### append set_idx_module_HarDBlock
            self.set_idx_module_HarDBlock.append( len(self.base)-1 )              # APPEND

            ### append module index (in ModuleList) into self.shortcut_layers
            if i_blk < (blks-1):
                self.shortcut_layers.append( len(self.base)-1 )
            #end


            ###------------------------------------------------------------------------------
            ### create transition-down layer
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], conv for ch_in:[%d]' % (i_blk,ch))
            self.base.append( ConvLayer(ch, ch_list[i_blk], kernel=1) )           # APPEND
            ch = ch_list[i_blk]
            #print('idx_blk:[%d], conv for ch_out:[%d]' % (i_blk,ch))


            ###------------------------------------------------------------------------------
            ### create AvgPool2d
            ###------------------------------------------------------------------------------
            if i_blk < blks-1:
                self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) )         # APPEND
            #end


            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<RPNet - backbone>>
            ###///////////////////////////////////////////////////////////////////////////////////////
            # print('idx_blk:[%d], ch_out_HarDBlock_this:[%d]' % (i_blk, ch_out_HarDBlock_this))
                # idx_blk: [0], ch_out_HarDBlock_this: [48]
                # idx_blk: [1], ch_out_HarDBlock_this: [78]
                # idx_blk: [2], ch_out_HarDBlock_this: [160]
                # idx_blk: [3], ch_out_HarDBlock_this: [214]
                # idx_blk: [4], ch_out_HarDBlock_this: [286]
        #end
        # completed to set
        #       <<for FC-HarDNet>>
        #         self.base[]
        #         self.shortcut_layers[]: having module index (in ModuleList) for shortcut (= here, skip_connection)
        #         skip_connection_channel_counts[]: having output ch of each HDB block (not transition-down layer)
        #       <<for RPNet-backbone>>
        #         self.set_idx_module_HarDBlock[]
        # ---------------------------------------------------------------------------------------------------------
        # note that
        #       skip_connection_channel_counts[] is for creating modules in the below (inside __init__()).
        #       self.shortcut_layers[] is used in forward().
        #---------------------------------------------------------------------------------------------------------
        # created HarDBlock & ConvL having
        #       idx_blk:[0], HarDBlock<ch_in: [48], ch_out: [48]>, ConvL<ch_in: [48], ch_out: [64]>
        #       idx_blk:[1], HarDBlock<ch_in: [64], ch_out: [78]>, ConvL<ch_in: [78], ch_out: [96]>
        #       idx_blk:[2], HarDBlock<ch_in: [96], ch_out: [160]>, ConvL<ch_in: [160], ch_out: [160]>
        #       idx_blk:[3], HarDBlock<ch_in: [160], ch_out: [214]>, ConvL<ch_in: [214], ch_out: [224]>
        #       idx_blk:[4], HarDBlock<ch_in: [224], ch_out: [286]>, ConvL<ch_in: [286], ch_out: [320]>
        #---------------------------------------------------------------------------------------------------------
        # To see the whole structure,
        #       please see FC_hardnet at /home/yu1/Desktop/avin_seg/seg_hardnet/
        #---------------------------------------------------------------------------------------------------------
        # message (2020/5/25)
        #   please, check shortcut
        #---------------------------------------------------------------------------------------------------------


        ###================================================================================================
        ### up-network
        ###================================================================================================
        cur_channels_count  = ch            # cur_channels_count: 320
        prev_block_channels = ch            # prev_block_channels: 320
        n_blocks            = blks-1        # n_blocks: 4, blks: 5
        self.n_blocks       = n_blocks

        ###
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])

        ###
        for i in range(n_blocks-1,-1,-1):
            #print('-'*50)
            #print('i:[%d]' % i)

            ###------------------------------------------------------------------------------
            ### append transition-up in self.transUpBlocks[]
            ###------------------------------------------------------------------------------
            self.transUpBlocks.append( TransitionUp(prev_block_channels, prev_block_channels) )         # APPEND
            #print('  prev_block_channels:[%d]' % prev_block_channels)


            ###------------------------------------------------------------------------------
            ### append conv1x1 in self.conv1x1_up[]
            ###------------------------------------------------------------------------------
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append( ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1) )    # APPEND
            #print('  cur_channels_count:[%d], skip_connection_channel_counts:[%d]' % (cur_channels_count, skip_connection_channel_counts[i]))
            #print('  cur_channels_count//2:[%d]' % (cur_channels_count//2))


            ###------------------------------------------------------------------------------
            ### append HarDBlock in self.denseBlocksUp[]
            ###------------------------------------------------------------------------------
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)                                      # APPEND


            ###------------------------------------------------------------------------------
            ### shift
            ###------------------------------------------------------------------------------
            prev_block_channels = blk.get_out_ch()
            #print('  [in] cur_channels_count: [%d]' % cur_channels_count)
            #print('  [out] blk.get_out_ch():[%d]' % prev_block_channels)

            cur_channels_count = prev_block_channels
        #end


        ###================================================================================================
        ### final conv (for FC-HarDNet)
        ###================================================================================================
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)


        ###================================================================================================
        ### relu for outcome of final conv (for rpnet)
        ###================================================================================================
        self.relu_on_finalConv = nn.ReLU(inplace=True)


        ###================================================================================================
        ### rpnet_decoder_cnvs (for rpnet)
        ###================================================================================================
        #ch_in_rpnet_decoder_cnvs = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b
        #self.rpnet_decoder_cnvs = ConvLayer(in_channels=ch_in_rpnet_decoder_cnvs, out_channels=256, kernel=3)


        ###================================================================================================
        ### rpnet_decoder_hmap_center (for rpnet)
        ###================================================================================================
        ch_in_rpnet_decoder_centerline = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b

        self.rpnet_decoder_centerline = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                  out_channels=1, kernel_size=1, stride=1,
                                                  padding=0, bias=True)

        self.rpnet_decoder_leftright  = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                  out_channels=2, kernel_size=1, stride=1,
                                                  padding=0, bias=True)



        #------------------------------------------------------------------------------
        # Through the above, the followings are set:
        #   self.base
        #   self.transUpBlocks
        #   self.denseBlocksUp
        #   self.conv1x1_up
        #   self.finalConv
        #
        #   self.rpnet_decoder_hmap_center
        #------------------------------------------------------------------------------

        #a = 1
    #end

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    ### hardnet::forward()
    ############################################################################################################
    def forward(self, x):

        ###================================================================================================
        ### init
        ###================================================================================================

        ###
        skip_connections = []
        size_in = x.size()


        ###================================================================================================
        ### down-network
        ###================================================================================================
        for idx_module in range(len(self.base)):
            ###------------------------------------------------------------------------------
            ###
            ###------------------------------------------------------------------------------
            x = self.base[idx_module](x)


            ###------------------------------------------------------------------------------
            ### (For FC-HarDNet)
            ###------------------------------------------------------------------------------
            if idx_module in self.shortcut_layers:
                skip_connections.append(x)
            #end
        #end


        ###================================================================================================
        ### up-network (for FC-HarDNet)
        ###================================================================================================
        out_seg = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out_seg = self.transUpBlocks[i](out_seg, skip, True)
            out_seg = self.conv1x1_up[i](out_seg)
            out_seg = self.denseBlocksUp[i](out_seg)
        #end


        ###================================================================================================
        ### insert semantic segmentation output (HDB-3U outcome) to backbone_rpnet
        ###================================================================================================
        backbone_rpnet = out_seg
            # completed to set
            #       backbone_rpnet (for rpnet)


        ###================================================================================================
        ### [FC-HarDNet] final conv
        ###================================================================================================
        out_seg = self.finalConv(out_seg)


        ###================================================================================================
        ### [rpnet] insert semantic segmentation output (final conv outcome) to backbone_rpnet
        ###================================================================================================
        out_seg_after_relu = self.relu_on_finalConv(out_seg)
        backbone_rpnet = torch.cat([backbone_rpnet, out_seg_after_relu], 1)
            # completed to set
            #       backbone_rpnet (for rpnet)


        ###================================================================================================
        ### [rpnet] rpnet_decoder_hmap_center
        ###================================================================================================
        out_centerline = self.rpnet_decoder_centerline(backbone_rpnet)
            # completed to set
            #       out_hmap_center

        out_leftright = self.rpnet_decoder_leftright(backbone_rpnet)
            # completed to set
            #       out_labelmap_left_right


        ###================================================================================================
        ### [FC-HarDNet] interpolate for final result
        ###================================================================================================
        out_seg_final = F.interpolate(out_seg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
            # completed to set
            #       out_seg_final


        ###================================================================================================
        ### [rpnet] interpolate for final result
        ###================================================================================================
        out_centerline_final = F.interpolate(out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
            # completed to set
            #       out_hmap_center_final

        # out_leftright_final = F.interpolate(out_leftright, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
            # completed to set
            #       out_labelmap_left_right_final



        return out_seg_final, out_centerline_final
    #end


########################################################################################################################
### main()
########################################################################################################################
if __name__ == '__main__':
    ###
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
    #end


    ###
    net = TPEnet_censeg(n_classes=3).cuda()


    ###
    with torch.no_grad():
        out_seg, out_centerline, out_leftright = net(torch.randn(2, 3, 540, 960).cuda())
        #y = net(torch.randn(2, 3, 512, 512).cuda())
        #y = net(torch.randn(2, 3, 1024, 512).cuda())
        print("out_seg.size()")
        print(out_seg.size())
        print("out_centerline.size()")
        print(out_centerline.size())
        print("out_labelmap_leftright.size()")
        print(out_leftright.size())
    #end


#end

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################




