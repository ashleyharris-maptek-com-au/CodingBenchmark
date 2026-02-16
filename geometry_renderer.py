"""
Vulkan Geometry Shader Renderer

Offscreen Vulkan renderer with full geometry shader support.
Uses the 'vulkan' Python package (cffi bindings to Vulkan C API).
Used by tests 42+ for geometry shader testing.

Vertex format: position(vec3) + normal(vec3) + color(vec3) = 36 bytes/vertex
"""

import numpy as np
import vulkan as vk
from vulkan import ffi as vk_ffi

# ---------------------------------------------------------------------------
# Topology mapping
# ---------------------------------------------------------------------------

TOPOLOGY_MAP = {
  "points": vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
  "lines": vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
  "triangles": vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  "line_adjacency": vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
  "triangle_adjacency": vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
}

VERTEX_STRIDE = 36  # 3*4 + 3*4 + 3*4 = 36 bytes


class VulkanGeometryRenderer:
  """Offscreen Vulkan renderer supporting geometry shaders."""

  def __init__(self, width=512, height=512):
    self.width = width
    self.height = height

    self._create_instance()
    self._pick_physical_device()
    self._create_device()
    self._create_command_pool()
    self._create_render_targets()
    self._create_render_pass()
    self._create_framebuffer()
    self._create_ubo()
    self._create_descriptor_resources()
    self._create_staging_buffer()

  # -------------------------------------------------------------------
  # Instance & Device
  # -------------------------------------------------------------------

  def _create_instance(self):
    app_info = vk.VkApplicationInfo(
      sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
      pApplicationName="GeomShaderTest",
      applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
      pEngineName="Test",
      engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
      apiVersion=vk.VK_API_VERSION_1_0,
    )
    create_info = vk.VkInstanceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      pApplicationInfo=app_info,
    )
    self.instance = vk.vkCreateInstance(create_info, None)

  def _pick_physical_device(self):
    devices = vk.vkEnumeratePhysicalDevices(self.instance)
    for pd in devices:
      features = vk.vkGetPhysicalDeviceFeatures(pd)
      if features.geometryShader:
        self.phys_dev = pd
        self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(pd)
        return
    raise RuntimeError("No GPU with geometry shader support found")

  def _find_memory_type(self, type_filter, properties):
    for i in range(self.mem_props.memoryTypeCount):
      if (type_filter & (1 << i)) and \
         (self.mem_props.memoryTypes[i].propertyFlags & properties) == properties:
        return i
    raise RuntimeError("Failed to find suitable memory type")

  def _create_device(self):
    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.phys_dev)
    self.graphics_family = 0
    for i, qf in enumerate(queue_families):
      if qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
        self.graphics_family = i
        break

    enabled_features = vk.VkPhysicalDeviceFeatures(geometryShader=vk.VK_TRUE)
    queue_create = vk.VkDeviceQueueCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      queueFamilyIndex=self.graphics_family,
      queueCount=1,
      pQueuePriorities=[1.0],
    )
    device_create = vk.VkDeviceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      queueCreateInfoCount=1,
      pQueueCreateInfos=[queue_create],
      pEnabledFeatures=enabled_features,
    )
    self.device = vk.vkCreateDevice(self.phys_dev, device_create, None)
    self.queue = vk.vkGetDeviceQueue(self.device, self.graphics_family, 0)

  def _create_command_pool(self):
    pool_info = vk.VkCommandPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      queueFamilyIndex=self.graphics_family,
      flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    )
    self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

    alloc_info = vk.VkCommandBufferAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool=self.command_pool,
      level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      commandBufferCount=1,
    )
    self.cmd_buf = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

  # -------------------------------------------------------------------
  # Helper: create buffer + memory
  # -------------------------------------------------------------------

  def _create_buffer(self, size, usage, mem_properties):
    buf_info = vk.VkBufferCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      size=size,
      usage=usage,
      sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
    )
    buf = vk.vkCreateBuffer(self.device, buf_info, None)
    mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buf)
    alloc_info = vk.VkMemoryAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      allocationSize=mem_reqs.size,
      memoryTypeIndex=self._find_memory_type(mem_reqs.memoryTypeBits, mem_properties),
    )
    mem = vk.vkAllocateMemory(self.device, alloc_info, None)
    vk.vkBindBufferMemory(self.device, buf, mem, 0)
    return buf, mem, mem_reqs.size

  def _upload_to_buffer(self, memory, data_bytes, size):
    mapped = vk.vkMapMemory(self.device, memory, 0, size, 0)
    vk_ffi.memmove(mapped, data_bytes, len(data_bytes))
    vk.vkUnmapMemory(self.device, memory)

  # -------------------------------------------------------------------
  # Helper: create image + memory
  # -------------------------------------------------------------------

  def _create_image(self, width, height, fmt, usage, aspect):
    img_info = vk.VkImageCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      imageType=vk.VK_IMAGE_TYPE_2D,
      format=fmt,
      extent=vk.VkExtent3D(width=width, height=height, depth=1),
      mipLevels=1,
      arrayLayers=1,
      samples=vk.VK_SAMPLE_COUNT_1_BIT,
      tiling=vk.VK_IMAGE_TILING_OPTIMAL,
      usage=usage,
      sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
      initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
    )
    image = vk.vkCreateImage(self.device, img_info, None)
    mem_reqs = vk.vkGetImageMemoryRequirements(self.device, image)
    alloc_info = vk.VkMemoryAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      allocationSize=mem_reqs.size,
      memoryTypeIndex=self._find_memory_type(mem_reqs.memoryTypeBits,
                                             vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    )
    mem = vk.vkAllocateMemory(self.device, alloc_info, None)
    vk.vkBindImageMemory(self.device, image, mem, 0)

    view_info = vk.VkImageViewCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      image=image,
      viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
      format=fmt,
      components=vk.VkComponentMapping(
        r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
      ),
      subresourceRange=vk.VkImageSubresourceRange(
        aspectMask=aspect,
        baseMipLevel=0,
        levelCount=1,
        baseArrayLayer=0,
        layerCount=1,
      ),
    )
    view = vk.vkCreateImageView(self.device, view_info, None)
    return image, mem, view

  # -------------------------------------------------------------------
  # Render targets
  # -------------------------------------------------------------------

  def _create_render_targets(self):
    self.color_format = vk.VK_FORMAT_R8G8B8A8_UNORM
    self.depth_format = vk.VK_FORMAT_D32_SFLOAT

    self.color_image, self.color_mem, self.color_view = self._create_image(
      self.width,
      self.height,
      self.color_format,
      vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      vk.VK_IMAGE_ASPECT_COLOR_BIT,
    )
    self.depth_image, self.depth_mem, self.depth_view = self._create_image(
      self.width,
      self.height,
      self.depth_format,
      vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      vk.VK_IMAGE_ASPECT_DEPTH_BIT,
    )

  # -------------------------------------------------------------------
  # Render pass
  # -------------------------------------------------------------------

  def _create_render_pass(self):
    color_attachment = vk.VkAttachmentDescription(
      format=self.color_format,
      samples=vk.VK_SAMPLE_COUNT_1_BIT,
      loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
      storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
      stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
      initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
      finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    )
    depth_attachment = vk.VkAttachmentDescription(
      format=self.depth_format,
      samples=vk.VK_SAMPLE_COUNT_1_BIT,
      loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
      storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
      stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
      initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
      finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )
    color_ref = vk.VkAttachmentReference(
      attachment=0,
      layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    )
    depth_ref = vk.VkAttachmentReference(
      attachment=1,
      layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )
    subpass = vk.VkSubpassDescription(
      pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
      colorAttachmentCount=1,
      pColorAttachments=[color_ref],
      pDepthStencilAttachment=depth_ref,
    )
    dependency = vk.VkSubpassDependency(
      srcSubpass=vk.VK_SUBPASS_EXTERNAL,
      dstSubpass=0,
      srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
      | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      srcAccessMask=0,
      dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
      | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
      | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    )
    rp_info = vk.VkRenderPassCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      attachmentCount=2,
      pAttachments=[color_attachment, depth_attachment],
      subpassCount=1,
      pSubpasses=[subpass],
      dependencyCount=1,
      pDependencies=[dependency],
    )
    self.render_pass = vk.vkCreateRenderPass(self.device, rp_info, None)

  # -------------------------------------------------------------------
  # Framebuffer
  # -------------------------------------------------------------------

  def _create_framebuffer(self):
    fb_info = vk.VkFramebufferCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      renderPass=self.render_pass,
      attachmentCount=2,
      pAttachments=[self.color_view, self.depth_view],
      width=self.width,
      height=self.height,
      layers=1,
    )
    self.framebuffer = vk.vkCreateFramebuffer(self.device, fb_info, None)

  # -------------------------------------------------------------------
  # UBO (same layout as test 41 for consistency)
  # -------------------------------------------------------------------

  def _create_ubo(self):
    from shader_test_utils import build_default_ubo, UBO_SIZE
    self.ubo_size = UBO_SIZE
    self.ubo_data = build_default_ubo(self.width, self.height)

    self.ubo_buf, self.ubo_mem, self.ubo_alloc_size = self._create_buffer(
      self.ubo_size,
      vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    self._upload_to_buffer(self.ubo_mem, self.ubo_data, self.ubo_alloc_size)

  # -------------------------------------------------------------------
  # Descriptor set layout, pool, set
  # -------------------------------------------------------------------

  def _create_descriptor_resources(self):
    binding = vk.VkDescriptorSetLayoutBinding(
      binding=0,
      descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount=1,
      stageFlags=(vk.VK_SHADER_STAGE_VERTEX_BIT | vk.VK_SHADER_STAGE_GEOMETRY_BIT
                  | vk.VK_SHADER_STAGE_FRAGMENT_BIT),
    )
    layout_info = vk.VkDescriptorSetLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount=1,
      pBindings=[binding],
    )
    self.desc_set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

    pool_size = vk.VkDescriptorPoolSize(
      type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount=1,
    )
    pool_info = vk.VkDescriptorPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      maxSets=1,
      poolSizeCount=1,
      pPoolSizes=[pool_size],
    )
    self.desc_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

    alloc_info = vk.VkDescriptorSetAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool=self.desc_pool,
      descriptorSetCount=1,
      pSetLayouts=[self.desc_set_layout],
    )
    self.desc_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

    buf_info = vk.VkDescriptorBufferInfo(
      buffer=self.ubo_buf,
      offset=0,
      range=self.ubo_size,
    )
    write = vk.VkWriteDescriptorSet(
      sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet=self.desc_set,
      dstBinding=0,
      dstArrayElement=0,
      descriptorCount=1,
      descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      pBufferInfo=[buf_info],
    )
    vk.vkUpdateDescriptorSets(self.device, 1, [write], 0, None)

    pl_info = vk.VkPipelineLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount=1,
      pSetLayouts=[self.desc_set_layout],
    )
    self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pl_info, None)

  # -------------------------------------------------------------------
  # Staging buffer for readback
  # -------------------------------------------------------------------

  def _create_staging_buffer(self):
    pixel_bytes = self.width * self.height * 4
    self.staging_buf, self.staging_mem, self.staging_alloc_size = self._create_buffer(
      pixel_bytes,
      vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )

  # -------------------------------------------------------------------
  # Render
  # -------------------------------------------------------------------

  def render(self,
             vert_spv,
             geom_spv,
             frag_spv,
             vertex_data_bytes,
             vertex_count,
             topology_name,
             index_data_bytes=None,
             index_count=0):
    """
        Render with geometry shader pipeline.

        Args:
            vert_spv: Vertex shader SPIR-V binary bytes
            geom_spv: Geometry shader SPIR-V binary bytes
            frag_spv: Fragment shader SPIR-V binary bytes
            vertex_data_bytes: Raw vertex buffer bytes (pos+normal+color, 36B/vert)
            vertex_count: Number of vertices
            topology_name: One of "points", "lines", "triangles",
                           "line_adjacency", "triangle_adjacency"
            index_data_bytes: Optional index buffer bytes (uint32)
            index_count: Number of indices (if using index buffer)

        Returns:
            numpy array (H, W, 4) uint8 RGBA
        """
    topology = TOPOLOGY_MAP[topology_name]

    # Create shader modules
    vert_mod = self._create_shader_module(vert_spv)
    geom_mod = self._create_shader_module(geom_spv)
    frag_mod = self._create_shader_module(frag_spv)

    # Create vertex buffer
    vb, vb_mem, vb_size = self._create_buffer(
      len(vertex_data_bytes),
      vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    self._upload_to_buffer(vb_mem, vertex_data_bytes, vb_size)

    # Optional index buffer
    ib = None
    ib_mem = None
    if index_data_bytes and index_count > 0:
      ib, ib_mem, ib_size = self._create_buffer(
        len(index_data_bytes),
        vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      )
      self._upload_to_buffer(ib_mem, index_data_bytes, ib_size)

    # Create pipeline
    pipeline = self._create_pipeline(vert_mod, geom_mod, frag_mod, topology)

    # Record command buffer
    self._record_commands(pipeline, vb, vertex_count, ib, index_count)

    # Submit
    submit_info = vk.VkSubmitInfo(
      sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
      commandBufferCount=1,
      pCommandBuffers=[self.cmd_buf],
    )
    vk.vkQueueSubmit(self.queue, 1, [submit_info], None)
    vk.vkQueueWaitIdle(self.queue)

    # Read back pixels
    pixels = self._readback()

    # Cleanup per-render resources
    vk.vkDestroyPipeline(self.device, pipeline, None)
    vk.vkDestroyShaderModule(self.device, vert_mod, None)
    vk.vkDestroyShaderModule(self.device, geom_mod, None)
    vk.vkDestroyShaderModule(self.device, frag_mod, None)
    vk.vkDestroyBuffer(self.device, vb, None)
    vk.vkFreeMemory(self.device, vb_mem, None)
    if ib is not None:
      vk.vkDestroyBuffer(self.device, ib, None)
      vk.vkFreeMemory(self.device, ib_mem, None)

    return pixels

  def _create_shader_module(self, spv_bytes):
    create_info = vk.VkShaderModuleCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      codeSize=len(spv_bytes),
      pCode=spv_bytes,
    )
    return vk.vkCreateShaderModule(self.device, create_info, None)

  def _create_pipeline(self, vert_mod, geom_mod, frag_mod, topology):
    # Shader stages
    vert_stage = vk.VkPipelineShaderStageCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
      module=vert_mod,
      pName="main",
    )
    geom_stage = vk.VkPipelineShaderStageCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage=vk.VK_SHADER_STAGE_GEOMETRY_BIT,
      module=geom_mod,
      pName="main",
    )
    frag_stage = vk.VkPipelineShaderStageCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
      module=frag_mod,
      pName="main",
    )

    # Vertex input: position(vec3) + normal(vec3) + color(vec3)
    binding_desc = vk.VkVertexInputBindingDescription(
      binding=0,
      stride=VERTEX_STRIDE,
      inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
    )
    attr_descs = [
      vk.VkVertexInputAttributeDescription(location=0,
                                           binding=0,
                                           format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                                           offset=0),
      vk.VkVertexInputAttributeDescription(location=1,
                                           binding=0,
                                           format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                                           offset=12),
      vk.VkVertexInputAttributeDescription(location=2,
                                           binding=0,
                                           format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                                           offset=24),
    ]
    vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      vertexBindingDescriptionCount=1,
      pVertexBindingDescriptions=[binding_desc],
      vertexAttributeDescriptionCount=3,
      pVertexAttributeDescriptions=attr_descs,
    )

    input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      topology=topology,
      primitiveRestartEnable=vk.VK_FALSE,
    )

    viewport = vk.VkViewport(
      x=0.0,
      y=0.0,
      width=float(self.width),
      height=float(self.height),
      minDepth=0.0,
      maxDepth=1.0,
    )
    scissor = vk.VkRect2D(
      offset=vk.VkOffset2D(x=0, y=0),
      extent=vk.VkExtent2D(width=self.width, height=self.height),
    )
    viewport_state = vk.VkPipelineViewportStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      viewportCount=1,
      pViewports=[viewport],
      scissorCount=1,
      pScissors=[scissor],
    )

    rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      depthClampEnable=vk.VK_FALSE,
      rasterizerDiscardEnable=vk.VK_FALSE,
      polygonMode=vk.VK_POLYGON_MODE_FILL,
      cullMode=vk.VK_CULL_MODE_NONE,
      frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
      depthBiasEnable=vk.VK_FALSE,
      lineWidth=1.0,
    )

    multisample = vk.VkPipelineMultisampleStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
      sampleShadingEnable=vk.VK_FALSE,
    )

    depth_stencil = vk.VkPipelineDepthStencilStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      depthTestEnable=vk.VK_TRUE,
      depthWriteEnable=vk.VK_TRUE,
      depthCompareOp=vk.VK_COMPARE_OP_LESS,
      depthBoundsTestEnable=vk.VK_FALSE,
      stencilTestEnable=vk.VK_FALSE,
    )

    color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
      blendEnable=vk.VK_FALSE,
      colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT
                      | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT),
    )
    color_blend = vk.VkPipelineColorBlendStateCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      logicOpEnable=vk.VK_FALSE,
      attachmentCount=1,
      pAttachments=[color_blend_attachment],
    )

    pipeline_info = vk.VkGraphicsPipelineCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      stageCount=3,
      pStages=[vert_stage, geom_stage, frag_stage],
      pVertexInputState=vertex_input,
      pInputAssemblyState=input_assembly,
      pViewportState=viewport_state,
      pRasterizationState=rasterizer,
      pMultisampleState=multisample,
      pDepthStencilState=depth_stencil,
      pColorBlendState=color_blend,
      layout=self.pipeline_layout,
      renderPass=self.render_pass,
      subpass=0,
    )
    pipelines = vk.vkCreateGraphicsPipelines(self.device, None, 1, [pipeline_info], None)
    return pipelines[0]

  def _record_commands(self, pipeline, vb, vertex_count, ib, index_count):
    begin_info = vk.VkCommandBufferBeginInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    )
    vk.vkBeginCommandBuffer(self.cmd_buf, begin_info)

    clear_values = [
      vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0])),
      vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)),
    ]
    rp_begin = vk.VkRenderPassBeginInfo(
      sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      renderPass=self.render_pass,
      framebuffer=self.framebuffer,
      renderArea=vk.VkRect2D(
        offset=vk.VkOffset2D(x=0, y=0),
        extent=vk.VkExtent2D(width=self.width, height=self.height),
      ),
      clearValueCount=2,
      pClearValues=clear_values,
    )
    vk.vkCmdBeginRenderPass(self.cmd_buf, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)

    vk.vkCmdBindPipeline(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
    vk.vkCmdBindDescriptorSets(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                               self.pipeline_layout, 0, 1, [self.desc_set], 0, None)
    vk.vkCmdBindVertexBuffers(self.cmd_buf, 0, 1, [vb], [0])

    if ib is not None and index_count > 0:
      vk.vkCmdBindIndexBuffer(self.cmd_buf, ib, 0, vk.VK_INDEX_TYPE_UINT32)
      vk.vkCmdDrawIndexed(self.cmd_buf, index_count, 1, 0, 0, 0)
    else:
      vk.vkCmdDraw(self.cmd_buf, vertex_count, 1, 0, 0)

    vk.vkCmdEndRenderPass(self.cmd_buf)

    # Copy color image to staging buffer
    region = vk.VkBufferImageCopy(
      bufferOffset=0,
      bufferRowLength=0,
      bufferImageHeight=0,
      imageSubresource=vk.VkImageSubresourceLayers(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevel=0,
        baseArrayLayer=0,
        layerCount=1,
      ),
      imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
      imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
    )
    vk.vkCmdCopyImageToBuffer(
      self.cmd_buf,
      self.color_image,
      vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      self.staging_buf,
      1,
      [region],
    )

    vk.vkEndCommandBuffer(self.cmd_buf)

  def _readback(self):
    pixel_bytes = self.width * self.height * 4
    mapped = vk.vkMapMemory(self.device, self.staging_mem, 0, self.staging_alloc_size, 0)
    data = bytes(mapped[0:pixel_bytes])
    vk.vkUnmapMemory(self.device, self.staging_mem)
    return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)

  # -------------------------------------------------------------------
  # Cleanup
  # -------------------------------------------------------------------

  def destroy(self):
    vk.vkDeviceWaitIdle(self.device)
    vk.vkDestroyBuffer(self.device, self.staging_buf, None)
    vk.vkFreeMemory(self.device, self.staging_mem, None)
    vk.vkDestroyBuffer(self.device, self.ubo_buf, None)
    vk.vkFreeMemory(self.device, self.ubo_mem, None)
    vk.vkDestroyDescriptorPool(self.device, self.desc_pool, None)
    vk.vkDestroyDescriptorSetLayout(self.device, self.desc_set_layout, None)
    vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
    vk.vkDestroyFramebuffer(self.device, self.framebuffer, None)
    vk.vkDestroyRenderPass(self.device, self.render_pass, None)
    vk.vkDestroyImageView(self.device, self.color_view, None)
    vk.vkDestroyImage(self.device, self.color_image, None)
    vk.vkFreeMemory(self.device, self.color_mem, None)
    vk.vkDestroyImageView(self.device, self.depth_view, None)
    vk.vkDestroyImage(self.device, self.depth_image, None)
    vk.vkFreeMemory(self.device, self.depth_mem, None)
    vk.vkDestroyCommandPool(self.device, self.command_pool, None)
    vk.vkDestroyDevice(self.device, None)
    vk.vkDestroyInstance(self.instance, None)


# ---------------------------------------------------------------------------
# Input Geometry Generators
# ---------------------------------------------------------------------------


def generate_point_grid(rows=5, cols=5, z=0.0):
  """Generate a grid of points in clip space [-0.8, 0.8].
    Returns (vertex_data_bytes, vertex_count)."""
  verts = []
  for r in range(rows):
    for c in range(cols):
      x = -0.8 + 1.6 * c / max(cols - 1, 1)
      y = -0.8 + 1.6 * r / max(rows - 1, 1)
      # position
      verts.extend([x, y, z])
      # normal (facing camera)
      verts.extend([0.0, 0.0, 1.0])
      # color (varies by position)
      cr = 0.5 + 0.5 * x
      cg = 0.5 + 0.5 * y
      cb = 0.3
      verts.extend([cr, cg, cb])
  data = np.array(verts, dtype=np.float32).tobytes()
  return data, rows * cols


def generate_line_segments(count=8):
  """Generate radial line segments from center.
    Returns (vertex_data_bytes, vertex_count)."""
  import math
  verts = []
  for i in range(count):
    angle = 2.0 * math.pi * i / count
    # Start near center
    x0, y0 = 0.1 * math.cos(angle), 0.1 * math.sin(angle)
    # End at edge
    x1, y1 = 0.8 * math.cos(angle), 0.8 * math.sin(angle)
    # vertex 0
    verts.extend([x0, y0, 0.0])
    verts.extend([0.0, 0.0, 1.0])
    r = 0.5 + 0.5 * math.cos(angle)
    g = 0.5 + 0.5 * math.sin(angle)
    verts.extend([r, g, 0.4])
    # vertex 1
    verts.extend([x1, y1, 0.0])
    verts.extend([0.0, 0.0, 1.0])
    verts.extend([r, g, 0.8])
  data = np.array(verts, dtype=np.float32).tobytes()
  return data, count * 2


def generate_line_adjacency_curve(num_segments=16):
  """Generate a smooth curve as line-adjacency primitives.
    Each segment needs 4 vertices: prev, start, end, next.
    Returns (vertex_data_bytes, vertex_count)."""
  import math
  # Generate curve control points (sine wave)
  num_points = num_segments + 1
  points = []
  for i in range(num_points + 2):  # +2 for adjacency at start/end
    t = (i - 1) / max(num_segments, 1)
    x = -0.8 + 1.6 * t
    y = 0.5 * math.sin(t * math.pi * 3.0)
    points.append((x, y, 0.0))

  verts = []
  for i in range(num_segments):
    # prev, start, end, next
    for j in [i, i + 1, i + 2, i + 3]:
      idx = max(0, min(j, len(points) - 1))
      px, py, pz = points[idx]
      verts.extend([px, py, pz])
      verts.extend([0.0, 0.0, 1.0])
      t = idx / max(len(points) - 1, 1)
      verts.extend([t, 0.5, 1.0 - t])
  data = np.array(verts, dtype=np.float32).tobytes()
  return data, num_segments * 4


def generate_sphere_triangles(stacks=16, sectors=32):
  """Generate sphere as triangle list with pos+normal+color format.
    Returns (vertex_data_bytes, vertex_count, index_data_bytes, index_count)."""
  import math
  verts = []
  for i in range(stacks + 1):
    phi = math.pi * i / stacks
    for j in range(sectors + 1):
      theta = 2.0 * math.pi * j / sectors
      x = math.sin(phi) * math.cos(theta)
      y = math.cos(phi)
      z = math.sin(phi) * math.sin(theta)
      # position (scaled to clip-ish space)
      verts.extend([x * 0.7, y * 0.7, z * 0.7])
      # normal
      verts.extend([x, y, z])
      # color
      verts.extend([0.5 + 0.5 * x, 0.5 + 0.5 * y, 0.5 + 0.5 * z])

  indices = []
  for i in range(stacks):
    for j in range(sectors):
      first = i * (sectors + 1) + j
      second = first + sectors + 1
      indices.extend([first, second, first + 1])
      indices.extend([second, second + 1, first + 1])

  vertex_data = np.array(verts, dtype=np.float32).tobytes()
  vertex_count = (stacks + 1) * (sectors + 1)
  index_data = np.array(indices, dtype=np.uint32).tobytes()
  index_count = len(indices)
  return vertex_data, vertex_count, index_data, index_count


def generate_triangle_adjacency_mesh():
  """Generate a simple quad mesh with triangle adjacency info.
    Returns (vertex_data_bytes, vertex_count, index_data_bytes, index_count)."""
  # Simple grid of quads, each split into 2 triangles
  # For adjacency, each triangle has 6 indices
  rows, cols = 4, 4
  # Generate vertices
  verts = []
  for r in range(rows + 1):
    for c in range(cols + 1):
      x = -0.8 + 1.6 * c / cols
      y = -0.8 + 1.6 * r / rows
      verts.extend([x, y, 0.0])
      verts.extend([0.0, 0.0, 1.0])
      cr = 0.3 + 0.5 * c / cols
      cg = 0.3 + 0.5 * r / rows
      verts.extend([cr, cg, 0.4])

  vertex_data = np.array(verts, dtype=np.float32).tobytes()
  vertex_count = (rows + 1) * (cols + 1)

  # Build triangle list with adjacency
  # For each quad (r,c), two triangles:
  # T1: (r*W+c, r*W+c+1, (r+1)*W+c)
  # T2: (r*W+c+1, (r+1)*W+c+1, (r+1)*W+c)
  W = cols + 1
  indices = []
  for r in range(rows):
    for c in range(cols):
      v00 = r * W + c
      v10 = r * W + c + 1
      v01 = (r + 1) * W + c
      v11 = (r + 1) * W + c + 1

      # Triangle 1: v00, v10, v01
      # adjacency: opposite v00 is v11, opposite v10 is v01-1, opposite v01 is v10+W
      # Simplified: use degenerate adjacency (self-referencing) for edge cases
      adj_01 = v11 if c < cols - 1 or r < rows - 1 else v00
      adj_12 = v11
      adj_20 = v00
      indices.extend([v00, adj_20, v10, adj_01, v01, adj_12])

      # Triangle 2: v10, v11, v01
      adj2_01 = v00
      adj2_12 = v00
      adj2_20 = v00
      indices.extend([v10, adj2_01, v11, adj2_12, v01, adj2_20])

  index_data = np.array(indices, dtype=np.uint32).tobytes()
  index_count = len(indices)
  return vertex_data, vertex_count, index_data, index_count
