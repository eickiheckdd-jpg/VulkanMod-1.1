package net.vulkanmod.vulkan.framebuffer;

import net.vulkanmod.vulkan.Renderer;
import net.vulkanmod.vulkan.VRenderSystem;
import net.vulkanmod.vulkan.Vulkan;
import net.vulkanmod.vulkan.memory.MemoryManager;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;

import static org.lwjgl.vulkan.KHRSwapchain.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
import static org.lwjgl.vulkan.VK10.*;

/**
 * A Vulkan 1.0-compatible render pass. Attachments are ordered color first,
 * then depth/stencil, and framebuffers are created to match that order.
 */
public class RenderPass {
    private Framebuffer framebuffer;
    private long id;

    private final int attachmentCount;
    private final AttachmentInfo colorAttachmentInfo;
    private final AttachmentInfo depthAttachmentInfo;

    public RenderPass(Framebuffer framebuffer, AttachmentInfo colorAttachmentInfo, AttachmentInfo depthAttachmentInfo) {
        this.framebuffer = framebuffer;
        this.colorAttachmentInfo = colorAttachmentInfo;
        this.depthAttachmentInfo = depthAttachmentInfo;
        this.attachmentCount = (colorAttachmentInfo == null ? 0 : 1)
                + (depthAttachmentInfo == null ? 0 : 1);

        if (attachmentCount == 0) {
            throw new IllegalArgumentException("A render pass must have at least one attachment");
        }

        createRenderPass();
    }

    private void createRenderPass() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkAttachmentDescription.Buffer attachments = VkAttachmentDescription.calloc(attachmentCount, stack);
            VkSubpassDescription.Buffer subpasses = VkSubpassDescription.calloc(1, stack);
            VkSubpassDescription subpass = subpasses.get(0);
            subpass.pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS);

            int attachmentIndex = 0;

            if (colorAttachmentInfo != null) {
                attachments.get(attachmentIndex)
                        .format(colorAttachmentInfo.format)
                        .samples(VK_SAMPLE_COUNT_1_BIT)
                        .loadOp(colorAttachmentInfo.loadOp)
                        .storeOp(colorAttachmentInfo.storeOp)
                        .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                        .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                        .initialLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        .finalLayout(colorAttachmentInfo.finalLayout);

                VkAttachmentReference.Buffer colorReference = VkAttachmentReference.calloc(1, stack);
                colorReference.get(0)
                        .attachment(attachmentIndex)
                        .layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                subpass.pColorAttachments(colorReference);
                subpass.colorAttachmentCount(1);
                attachmentIndex++;
            }

            if (depthAttachmentInfo != null) {
                attachments.get(attachmentIndex)
                        .format(depthAttachmentInfo.format)
                        .samples(VK_SAMPLE_COUNT_1_BIT)
                        .loadOp(depthAttachmentInfo.loadOp)
                        .storeOp(depthAttachmentInfo.storeOp)
                        .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                        .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                        .initialLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .finalLayout(depthAttachmentInfo.finalLayout);

                VkAttachmentReference depthReference = VkAttachmentReference.calloc(stack);
                depthReference.attachment(attachmentIndex)
                        .layout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                subpass.pDepthStencilAttachment(depthReference);
            }

            // Synchronize prior uses of the same attachments before this subpass.
            // Layout changes from sampled/transfer/present layouts are handled by
            // the explicit image barriers recorded immediately before the pass.
            int sourceStages = 0;
            int destinationStages = 0;
            int sourceAccess = 0;
            int destinationAccess = 0;

            if (colorAttachmentInfo != null) {
                sourceStages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                destinationStages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                sourceAccess |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                destinationAccess |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            }
            if (depthAttachmentInfo != null) {
                sourceStages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                        | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                destinationStages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                        | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                sourceAccess |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                destinationAccess |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                        | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }

            VkSubpassDependency.Buffer dependencies = VkSubpassDependency.calloc(2, stack);
            int dependencyCount = 0;
            dependencies.get(dependencyCount++)
                    .srcSubpass(VK_SUBPASS_EXTERNAL)
                    .dstSubpass(0)
                    .srcStageMask(sourceStages)
                    .dstStageMask(destinationStages)
                    .srcAccessMask(sourceAccess)
                    .dstAccessMask(destinationAccess)
                    .dependencyFlags(0);

            // Make attachment writes visible to subsequent sampling/presentation.
            int externalDestinationStages = 0;
            int externalSourceStages = 0;
            int externalSourceAccess = 0;
            int externalDestinationAccess = 0;

            if (colorAttachmentInfo != null) {
                if (colorAttachmentInfo.finalLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    externalSourceStages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                    externalDestinationStages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                    externalSourceAccess |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    externalDestinationAccess |= VK_ACCESS_SHADER_READ_BIT;
                } else if (colorAttachmentInfo.finalLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                    externalSourceStages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                    externalDestinationStages |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                    externalSourceAccess |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                }
            }

            if (depthAttachmentInfo != null
                    && depthAttachmentInfo.finalLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
                externalSourceStages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                        | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                externalDestinationStages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                externalSourceAccess |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                externalDestinationAccess |= VK_ACCESS_SHADER_READ_BIT;
            }

            if (externalSourceStages != 0) {
                dependencies.get(dependencyCount++)
                        .srcSubpass(0)
                        .dstSubpass(VK_SUBPASS_EXTERNAL)
                        .srcStageMask(externalSourceStages)
                        .dstStageMask(externalDestinationStages)
                        .srcAccessMask(externalSourceAccess)
                        .dstAccessMask(externalDestinationAccess)
                        .dependencyFlags(0);
            }

            VkRenderPassCreateInfo createInfo = VkRenderPassCreateInfo.calloc(stack);
            createInfo.sType$Default()
                    .pAttachments(attachments)
                    .pSubpasses(subpasses);
            if (dependencyCount > 0) {
                dependencies.limit(dependencyCount);
                createInfo.pDependencies(dependencies);
            }

            LongBuffer renderPassHandle = stack.mallocLong(1);
            Vulkan.checkResult(
                    vkCreateRenderPass(Vulkan.getVkDevice(), createInfo, null, renderPassHandle),
                    "Failed to create render pass"
            );
            this.id = renderPassHandle.get(0);
        }
    }

    /** Records the classic vkCmdBeginRenderPass command. */
    public void beginRenderPass(VkCommandBuffer commandBuffer, long framebufferId, MemoryStack stack) {
        if (colorAttachmentInfo != null
                && framebuffer.getColorAttachment().getCurrentLayout() != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            framebuffer.getColorAttachment().transitionImageLayout(
                    stack, commandBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
        if (depthAttachmentInfo != null
                && framebuffer.getDepthAttachment().getCurrentLayout() != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            framebuffer.getDepthAttachment().transitionImageLayout(
                    stack, commandBuffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }

        VkRenderPassBeginInfo beginInfo = VkRenderPassBeginInfo.calloc(stack);
        beginInfo.sType$Default()
                .renderPass(this.id)
                .framebuffer(framebufferId);

        VkRect2D renderArea = VkRect2D.calloc(stack);
        renderArea.offset().set(0, 0);
        renderArea.extent().set(framebuffer.getWidth(), framebuffer.getHeight());
        beginInfo.renderArea(renderArea);

        // Clear values are indexed by attachment description, not by attachment type.
        VkClearValue.Buffer clearValues = VkClearValue.calloc(attachmentCount, stack);
        int attachmentIndex = 0;
        if (colorAttachmentInfo != null) {
            clearValues.get(attachmentIndex++).color().float32(VRenderSystem.clearColor);
        }
        if (depthAttachmentInfo != null) {
            clearValues.get(attachmentIndex).depthStencil().set(1.0f, 0);
        }
        beginInfo.pClearValues(clearValues);

        vkCmdBeginRenderPass(commandBuffer, beginInfo, VK_SUBPASS_CONTENTS_INLINE);
        Renderer.getInstance().setBoundRenderPass(this);
    }

    public void endRenderPass(VkCommandBuffer commandBuffer) {
        vkCmdEndRenderPass(commandBuffer);

        // The render pass performs these final layout transitions implicitly.
        if (colorAttachmentInfo != null) {
            framebuffer.getColorAttachment().setCurrentLayout(colorAttachmentInfo.finalLayout);
        }
        if (depthAttachmentInfo != null) {
            framebuffer.getDepthAttachment().setCurrentLayout(depthAttachmentInfo.finalLayout);
        }

        Renderer.getInstance().setBoundRenderPass(null);
    }

    public Framebuffer getFramebuffer() {
        return framebuffer;
    }

    public void setFramebuffer(Framebuffer framebuffer) {
        this.framebuffer = framebuffer;
    }

    public boolean hasColorAttachment() {
        return colorAttachmentInfo != null;
    }

    public boolean hasDepthAttachment() {
        return depthAttachmentInfo != null;
    }

    public int getAttachmentCount() {
        return attachmentCount;
    }

    public void cleanUp() {
        long renderPassId = this.id;
        if (renderPassId == VK_NULL_HANDLE) {
            return;
        }

        this.id = VK_NULL_HANDLE;
        MemoryManager.getInstance().addFrameOp(
                () -> vkDestroyRenderPass(Vulkan.getVkDevice(), renderPassId, null)
        );
    }

    public long getId() {
        return id;
    }

    public static class AttachmentInfo {
        final Type type;
        final int format;
        int finalLayout;
        int loadOp;
        int storeOp;

        public AttachmentInfo(Type type, int format) {
            this.type = type;
            this.format = format;
            this.finalLayout = type.defaultLayout;
            this.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            this.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }

        public AttachmentInfo setOps(int loadOp, int storeOp) {
            this.loadOp = loadOp;
            this.storeOp = storeOp;
            return this;
        }

        public AttachmentInfo setLoadOp(int loadOp) {
            this.loadOp = loadOp;
            return this;
        }

        public AttachmentInfo setFinalLayout(int finalLayout) {
            this.finalLayout = finalLayout;
            return this;
        }

        public enum Type {
            COLOR(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
            DEPTH(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

            final int defaultLayout;

            Type(int layout) {
                this.defaultLayout = layout;
            }
        }
    }

    public static Builder builder(Framebuffer framebuffer) {
        return new Builder(framebuffer);
    }

    public static class Builder {
        private final Framebuffer framebuffer;
        private final AttachmentInfo colorAttachmentInfo;
        private final AttachmentInfo depthAttachmentInfo;

        public Builder(Framebuffer framebuffer) {
            this.framebuffer = framebuffer;

            this.colorAttachmentInfo = framebuffer.hasColorAttachment
                    ? new AttachmentInfo(AttachmentInfo.Type.COLOR, framebuffer.format)
                    .setOps(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE)
                    : null;
            this.depthAttachmentInfo = framebuffer.hasDepthAttachment
                    ? new AttachmentInfo(AttachmentInfo.Type.DEPTH, framebuffer.depthFormat)
                    .setOps(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE)
                    : null;
        }

        public RenderPass build() {
            return new RenderPass(framebuffer, colorAttachmentInfo, depthAttachmentInfo);
        }

        public Builder setLoadOp(int loadOp) {
            if (colorAttachmentInfo != null) {
                colorAttachmentInfo.setLoadOp(loadOp);
            }
            if (depthAttachmentInfo != null) {
                depthAttachmentInfo.setLoadOp(loadOp);
            }
            return this;
        }

        public AttachmentInfo getColorAttachmentInfo() {
            return colorAttachmentInfo;
        }

        public AttachmentInfo getDepthAttachmentInfo() {
            return depthAttachmentInfo;
        }
    }
}