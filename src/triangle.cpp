#include "common.h"

VkPipelineLayout createPipelineLayout(VkDevice device)
{
	VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layoutInfo.setLayoutCount = 0;
	layoutInfo.pushConstantRangeCount = 0;
	layoutInfo.pSetLayouts = nullptr;
	layoutInfo.pPushConstantRanges = nullptr;

	VkPipelineLayout layout = VK_NULL_HANDLE;
	VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));
	return layout;
}

void execute(char** argv)
{
    VK_CHECK(volkInitialize());

    int rc = glfwInit();
    assert(rc);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    VkInstance instance = createInstance();
    assert(instance);

    volkLoadInstanceOnly(instance);

    VkDebugReportCallbackEXT debugCallback = registerDebugCallback(instance);

    VkPhysicalDevice physicalDevices[16];
    uint32_t physicalDeviceCount =
        std::size(physicalDevices);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                        physicalDevices));

    VkPhysicalDevice physicalDevice =
        pickPhysicalDevice(physicalDevices, physicalDeviceCount);
    assert(physicalDevice);

    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0,
                                                  &extensionCount, 0));

    std::vector<VkExtensionProperties> extensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &extensionCount, extensions.data()));

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    assert(props.limits.timestampComputeAndGraphics);

    uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
    assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);

    VkDevice device =
        createDevice(instance, physicalDevice, familyIndex, false);
    assert(device);

    volkLoadDevice(device);

    vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    vkCmdEndRendering = vkCmdEndRenderingKHR;
    vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;

    GLFWwindow *window = glfwCreateWindow(1024, 768, "meshtrex", nullptr, nullptr);
    assert(window);

    VkSurfaceKHR surface = createSurface(instance, window);
    assert(surface);

    VkBool32 presentSupported = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex,
                                                  surface, &presentSupported));
    assert(presentSupported);

    VkFormat swapchainFormat = getSwapchainFormat(physicalDevice, surface);

    VkSemaphore acquireSemaphore = createSemaphore(device);
    assert(acquireSemaphore);

    VkSemaphore releaseSemaphore = createSemaphore(device);
    assert(releaseSemaphore);

    VkFence frameFence = createFence(device);
    assert(frameFence);

    VkQueue queue = 0;
    vkGetDeviceQueue(device, familyIndex, 0, &queue);

    Shader triangleVS {};
    assert(loadShader(triangleVS, device, argv[0], "spirv/triangle.vert.spv"));

    Shader triangleFS {};
    assert(loadShader(triangleFS, device, argv[0], "spirv/triangle.frag.spv"));

    // TODO: this is critical for performance!
    VkPipelineCache pipelineCache = nullptr;
	VkPipelineRenderingCreateInfo renderingInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &swapchainFormat,
		.depthAttachmentFormat = VK_FORMAT_UNDEFINED,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED
	};

    VkPipelineLayout pipelineLayout = createPipelineLayout(device);
    VkPipeline trianglePipeline = createGraphicsPipeline(
        device,
        pipelineCache,
        renderingInfo,
        {&triangleVS, &triangleFS},
        pipelineLayout,
        {}
        );

    Swapchain swapchain;
    createSwapchain(swapchain, physicalDevice, device, surface, familyIndex,
                    window, swapchainFormat);

	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    assert(commandPool);

    VkCommandBufferAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};

    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

	std::vector<VkImageView> swapchainImageViews(swapchain.imageCount);
    VkCommandBuffer commandBuffer = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

    while (!glfwWindowShouldClose(window)) {
    	static int frame = 0;
    	printf("Drawing frame: %d\n", frame++);
        glfwPollEvents();

    	uint32_t imageIndex = 0;
    	VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
    	if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    		continue;
    	VK_CHECK_SWAPCHAIN(acquireResult);

    	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    	VkImageMemoryBarrier barrierToRender = {
    		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapchain.images[imageIndex],
			.subresourceRange = {
    			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
    	};

    	vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrierToRender
		);

    	VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    	createInfo.image = swapchain.images[imageIndex];
    	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    	createInfo.format = getSwapchainFormat(physicalDevice, surface);
    	createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    	createInfo.subresourceRange.levelCount = 1;
    	createInfo.subresourceRange.layerCount = 1;

    	VkImageView view = nullptr;
    	VK_CHECK(vkCreateImageView(device, &createInfo, 0, &view));
    	VkRenderingAttachmentInfo colorAttachment = {
    		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = view,
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue =  { 48.f / 255.f, 10.f / 255.f, 36.f / 255.f, 1 }
		};

    	VkRenderingInfo passInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    	passInfo.renderArea.extent.width = swapchain.width;
    	passInfo.renderArea.extent.height = swapchain.height;
    	passInfo.layerCount = 1;
    	passInfo.colorAttachmentCount = 1;
    	passInfo.pColorAttachments = &colorAttachment;
    	// passInfo.pDepthAttachment = &depthAttachment;

		vkCmdBeginRendering(commandBuffer, &passInfo);

    	VkViewport viewport = {  0.0f,(float)swapchain.height, (float)swapchain.width,-(float)swapchain.height, 0.0f, 1.0f };
		VkRect2D scissor = { {0, 0}, {(swapchain.width), (swapchain.height)} };

		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);

		vkCmdEndRendering(commandBuffer);

    	VkImageMemoryBarrier barrierToPresent = barrierToRender;
    	barrierToPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    	barrierToPresent.dstAccessMask = 0;
    	barrierToPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    	barrierToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    	vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrierToPresent
		);

    	VK_CHECK(vkEndCommandBuffer(commandBuffer));

		VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &acquireSemaphore;
		submitInfo.pWaitDstStageMask = &submitStageMask;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &releaseSemaphore;

		VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &releaseSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapchain.swapchain;
		presentInfo.pImageIndices = &imageIndex;

		VK_CHECK(vkQueuePresentKHR(queue, &presentInfo));

		VK_CHECK(vkDeviceWaitIdle(device));

		glfwWaitEvents();
    }

    VK_CHECK(vkDeviceWaitIdle(device));

    vkDestroyCommandPool(device, commandPool, nullptr);

    destroySwapchain(device, swapchain);

	vkDestroyFence(device, frameFence, 0);
    vkDestroySemaphore(device, releaseSemaphore, nullptr);
    vkDestroySemaphore(device, acquireSemaphore, nullptr);

    vkDestroySurfaceKHR(instance, surface, 0);

    glfwDestroyWindow(window);

    vkDestroyDevice(device, nullptr);

	if (debugCallback)
		vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);

    vkDestroyInstance(instance, 0);

	volkFinalize();
}