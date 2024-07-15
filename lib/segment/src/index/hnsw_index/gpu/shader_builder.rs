use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_vector_storage::GpuVectorStorageElementType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutSetBinding {
    VisitedFlags,
    VectorStorage,
    Links,
    CandidatesHeap,
    NearestHeap,
}

impl LayoutSetBinding {
    pub fn to_string(self) -> &'static str {
        match self {
            LayoutSetBinding::VisitedFlags => "VISITED_FLAGS_LAYOUT_SET",
            LayoutSetBinding::VectorStorage => "VECTOR_STORAGE_LAYOUT_SET",
            LayoutSetBinding::Links => "LINKS_LAYOUT_SET",
            LayoutSetBinding::CandidatesHeap => "CANDIDATES_HEAP_LAYOUT_SET",
            LayoutSetBinding::NearestHeap => "NEAREST_HEAP_LAYOUT_SET",
        }
    }
}

pub struct ShaderBuilder {
    device: Arc<gpu::Device>,
    working_group_size: usize,
    shader_code: String,
    element_type: Option<GpuVectorStorageElementType>,
    layout_bindings: Vec<(LayoutSetBinding, usize)>,
    dim: Option<usize>,
    storages_count: Option<usize>,
    storage_size: Option<usize>,
    nearest_heap_ef: Option<usize>,
    nearest_heap_capacity: Option<usize>,
    candidates_heap_capacity: Option<usize>,
    links_capacity: Option<usize>,
    shaders_map: HashMap<String, String>,
}

impl ShaderBuilder {
    pub fn new(device: Arc<gpu::Device>, working_group_size: usize) -> Self {
        let shaders_map = HashMap::from([
            (
                "bheap.comp".to_string(),
                include_str!("shaders/bheap.comp").to_string(),
            ),
            (
                "candidates_heap.comp".to_string(),
                include_str!("shaders/candidates_heap.comp").to_string(),
            ),
            (
                "common.comp".to_string(),
                include_str!("shaders/common.comp").to_string(),
            ),
            (
                "iterators.comp".to_string(),
                include_str!("shaders/iterators.comp").to_string(),
            ),
            (
                "links.comp".to_string(),
                include_str!("shaders/links.comp").to_string(),
            ),
            (
                "nearest_heap.comp".to_string(),
                include_str!("shaders/nearest_heap.comp").to_string(),
            ),
            (
                "run_get_patch.comp".to_string(),
                include_str!("shaders/run_get_patch.comp").to_string(),
            ),
            (
                "run_greedy_search.comp".to_string(),
                include_str!("shaders/run_greedy_search.comp").to_string(),
            ),
            (
                "run_insert_vector.comp".to_string(),
                include_str!("shaders/run_insert_vector.comp").to_string(),
            ),
            (
                "search_context.comp".to_string(),
                include_str!("shaders/search_context.comp").to_string(),
            ),
            (
                "vector_storage.comp".to_string(),
                include_str!("shaders/vector_storage.comp").to_string(),
            ),
            (
                "visited_flags.comp".to_string(),
                include_str!("shaders/visited_flags.comp").to_string(),
            ),
        ]);

        Self {
            device,
            working_group_size,
            shader_code: Default::default(),
            element_type: None,
            layout_bindings: Default::default(),
            dim: None,
            storages_count: None,
            storage_size: None,
            nearest_heap_ef: None,
            nearest_heap_capacity: None,
            candidates_heap_capacity: None,
            links_capacity: None,
            shaders_map,
        }
    }

    pub fn with_shader_code(&mut self, shader_code: &str) -> &mut Self {
        self.shader_code.push_str("\n");
        self.shader_code.push_str(shader_code);
        self
    }

    pub fn with_element_type(&mut self, element_type: GpuVectorStorageElementType) -> &mut Self {
        self.element_type = Some(element_type);
        self
    }

    pub fn with_dim(&mut self, dim: usize) -> &mut Self {
        self.dim = Some(dim);
        self
    }

    pub fn with_storages_count(&mut self, storages_count: usize) -> &mut Self {
        self.storages_count = Some(storages_count);
        self
    }

    pub fn with_storage_size(&mut self, storage_size: usize) -> &mut Self {
        self.storage_size = Some(storage_size);
        self
    }

    pub fn with_nearest_heap_ef(&mut self, nearest_heap_ef: usize) -> &mut Self {
        self.nearest_heap_ef = Some(nearest_heap_ef);
        self
    }

    pub fn with_nearest_heap_capacity(&mut self, nearest_heap_capacity: usize) -> &mut Self {
        self.nearest_heap_capacity = Some(nearest_heap_capacity);
        self
    }

    pub fn with_candidates_heap_capacity(&mut self, candidates_heap_capacity: usize) -> &mut Self {
        self.candidates_heap_capacity = Some(candidates_heap_capacity);
        self
    }

    pub fn with_links_capacity(&mut self, links_capacity: usize) -> &mut Self {
        self.links_capacity = Some(links_capacity);
        self
    }

    pub fn with_layout(&mut self, layout: LayoutSetBinding, binding: usize) -> &mut Self {
        self.layout_bindings.push((layout, binding));
        self
    }

    pub fn build(&self) -> Arc<gpu::Shader> {
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_3);

        options.add_macro_definition(
            "WORKING_GROUP_SIZE",
            Some(&self.working_group_size.to_string()),
        );
        options.add_macro_definition(
            "SUBGROUP_SIZE",
            Some(&self.device.subgroup_size().to_string()),
        );

        if let Some(element_type) = self.element_type {
            match element_type {
                GpuVectorStorageElementType::Float32 => {
                    options.add_macro_definition("VECTOR_STORAGE_ELEMENT_FLOAT32", None)
                }
                GpuVectorStorageElementType::Float16 => {
                    options.add_macro_definition("VECTOR_STORAGE_ELEMENT_FLOAT16", None)
                }
                GpuVectorStorageElementType::Uint8 => {
                    options.add_macro_definition("VECTOR_STORAGE_ELEMENT_UINT8", None)
                }
                GpuVectorStorageElementType::Binary => {
                    options.add_macro_definition("VECTOR_STORAGE_ELEMENT_BINARY", None)
                }
            }
        }

        for (layout, binding) in &self.layout_bindings {
            options.add_macro_definition(layout.to_string(), Some(&binding.to_string()));
        }

        if let Some(dim) = self.dim {
            options.add_macro_definition("DIM", Some(&dim.to_string()));
        }

        if let Some(storages_count) = self.storages_count {
            options.add_macro_definition("STORAGES_COUNT", Some(&storages_count.to_string()));
        }

        if let Some(storage_size) = self.storage_size {
            options.add_macro_definition("STORAGE_SIZE", Some(&storage_size.to_string()));
        }

        if let Some(nearest_heap_ef) = self.nearest_heap_ef {
            options.add_macro_definition("NEAREST_HEAP_EF", Some(&nearest_heap_ef.to_string()));
        }

        if let Some(nearest_heap_capacity) = self.nearest_heap_capacity {
            options.add_macro_definition(
                "NEAREST_HEAP_CAPACITY",
                Some(&nearest_heap_capacity.to_string()),
            );
        }

        if let Some(candidates_heap_capacity) = self.candidates_heap_capacity {
            options.add_macro_definition(
                "CANDIDATES_HEAP_CAPACITY",
                Some(&candidates_heap_capacity.to_string()),
            );
        }

        if let Some(links_capacity) = self.links_capacity {
            options.add_macro_definition("LINKS_CAPACITY", Some(&links_capacity.to_string()));
        }

        options.set_include_callback(|filename, _, _, _| {
            let code = self.shaders_map.get(filename).unwrap();
            Ok(shaderc::ResolvedInclude {
                resolved_name: filename.to_string(),
                content: code.to_owned(),
            })
        });

        let timer = std::time::Instant::now();
        let compiled = self
            .device
            .compiler
            .compile_into_spirv(
                &self.shader_code,
                shaderc::ShaderKind::Compute,
                "shader.glsl",
                "main",
                Some(&options),
            )
            .unwrap();
        log::debug!("Shader compilation took: {:?}", timer.elapsed());
        Arc::new(gpu::Shader::new(
            self.device.clone(),
            compiled.as_binary_u8(),
        ))
    }
}
