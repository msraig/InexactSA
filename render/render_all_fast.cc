#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/tensor_format.h"
#include <string>


void renderImage(const std::string& instanceID,
				 float* output_diffuse_sum,
				 float* output_spec_sum, 
				 const float* albedo_data,
				 const float* spec_data,
				 const float* roughness_data,
				 const float* normal_data,
				 const float* mask_data,
				 const float* matLight_data,
				 const int* lightMapID,
				 const float* matWorld_data,
				 const float* matView_data,
				 const float* matProj_data,
				 const float* eyePosWorld_data,
				 const int imgHeight, const int imgWidth, const int batchSize,
				 const int nSampleDiffuse, const int nSampleSpec, const int nCubeRes, const bool useFresnel, const int renderMode);


void cameraSetup(const std::string& id,
                 const float* matWorld_data, const float* eyePosWorld_data, 
                 const float* matView_data, const float* matProj_data,
                 const int imgHeight, const int imgWidth);

void initLightMap(const std::string& id, const std::string& lightfolder, const std::string& lightfile, const int envWidth);
void freeRenderOp(const std::string& id);
void reportGPUMemory();

using namespace tensorflow;

REGISTER_OP("RenderShadingSum")
	.Attr("light_folder: string")
	.Attr("light_file: string")
	.Attr("num_sample_diffuse: int = 256")
	.Attr("num_sample_spec: int = 1024")
	.Attr("env_width: int = 256")
	.Attr("instance_id: string = 'render_0'")
	.Attr("use_fresnel: int = 1")
	.Attr("render_mode: int = 0")
    .Input("albedo: float")
    .Input("spec: float")
    .Input("roughness: float")
    .Input("normal: float")			//xy
    .Input("mask: float")
    .Input("light_id: int32")
    .Input("matlight: float")
	.Input("matworld: float")
	.Input("eye_pos_world: float")
	.Input("matview: float")
	.Input("matproj: float")
    .Output("diffuse_shading: float")
    .Output("spec_shading: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return Status::OK();
    });

//batch, height, width, channels

using GPUDevice = Eigen::GpuDevice;

class RenderShadingSumOp : public OpKernel {
	public:
		explicit RenderShadingSumOp(OpKernelConstruction* context) : OpKernel(context) {

			std::string light_folder, light_file;

			context->GetAttr("num_sample_diffuse", &nSampleDiffuse);
			context->GetAttr("num_sample_spec", &nSampleSpec);
			context->GetAttr("light_folder", &light_folder);
			context->GetAttr("light_file", &light_file);
			context->GetAttr("env_width", &envWidth);
			context->GetAttr("instance_id", &instanceID);
			context->GetAttr("use_fresnel", &useFresnel);
			context->GetAttr("render_mode", &renderMode);

			initLightMap(instanceID, light_folder, light_file, envWidth);
		}

		~RenderShadingSumOp()
		{
			freeRenderOp(instanceID);
		}

		void Compute(OpKernelContext* context) override {
    	// Grab the input tensor
    		const Tensor& input_albedo 		= context->input(0);
    		const Tensor& input_spec 		= context->input(1);
    		const Tensor& input_roughness 	= context->input(2);
    		const Tensor& input_normal 		= context->input(3);

    		const Tensor& input_mask		= context->input(4);

    		const Tensor& input_lightID 	= context->input(5);
    		const Tensor& input_matLight	= context->input(6);

    		const Tensor& input_matWorld	= context->input(7);
    		const Tensor& input_eyePosWorld	= context->input(8);
			const Tensor& input_matView 	= context->input(9);
    		const Tensor& input_matProj 	= context->input(10);			

    		this->img_height = (int)input_albedo.dim_size(1);
    		this->img_width = (int)input_albedo.dim_size(2);


    		Tensor* output_tensor_diffuse = NULL;
    		Tensor* output_tensor_spec = NULL;   		
    		OP_REQUIRES_OK(context, context->allocate_output(0, input_albedo.shape(),
													 &output_tensor_diffuse));
    		OP_REQUIRES_OK(context, context->allocate_output(1, input_albedo.shape(),
													 &output_tensor_spec));    													 		
    		bool fresnel = (useFresnel == 1);
			batchRender(output_tensor_diffuse, output_tensor_spec,
						&input_albedo, &input_spec, &input_roughness, &input_normal,
						&input_mask, 
						&input_lightID, &input_matLight,
						&input_matWorld, &input_matView, &input_matProj, &input_eyePosWorld,
						fresnel, renderMode);
		}


	private:
		int img_height;
		int img_width;
		int nSampleDiffuse;
		int nSampleSpec;
		int envWidth;

		int useFresnel;

		int renderMode;

		std::string instanceID;
//		Tensor lightMap;

		void batchRender(Tensor* rendered_diffuse_sum, Tensor* rendered_spec_sum, 
						 const Tensor* albedo, const Tensor* spec, const Tensor* roughness, const Tensor* normal,
						 const Tensor* mask,
						 const Tensor* lightMapID, const Tensor* matLight,
						 const Tensor* matWorld, const Tensor* matView, const Tensor* matProj, const Tensor* eyePosWorld,
						 const bool useFresnel, const int renderMode)
		{
			auto batch_size = rendered_diffuse_sum->dim_size(0);
		
			auto albedo_ptr 		= albedo->flat<float>().data();
			auto spec_ptr 			= spec->flat<float>().data();
			auto roughness_ptr 		= roughness->flat<float>().data();
			auto normal_ptr 		= normal->flat<float>().data();
			auto mask_ptr 			= mask->flat<float>().data();
			auto matWorld_ptr 		= matWorld->flat<float>().data();
			auto matView_ptr 		= matView->flat<float>().data();
			auto matProj_ptr 		= matProj->flat<float>().data();
			auto eyePosWorld_ptr 	= eyePosWorld->flat<float>().data();									
			auto matLight_ptr 		= matLight->flat<float>().data();
			auto lightID_ptr 		= lightMapID->flat<int>().data();

			auto rendered_diffuse_sum_ptr 	= rendered_diffuse_sum->flat<float>().data();
			auto rendered_spec_sum_ptr 		= rendered_spec_sum->flat<float>().data();

			renderImage(instanceID,
						rendered_diffuse_sum_ptr,
						rendered_spec_sum_ptr, 
						albedo_ptr, spec_ptr, roughness_ptr, normal_ptr, 
						mask_ptr,
						matLight_ptr, lightID_ptr, 
						matWorld_ptr, 
						matView_ptr,
						matProj_ptr,
						eyePosWorld_ptr,
						img_height, img_width, batch_size,
						nSampleDiffuse, nSampleSpec, envWidth, useFresnel, renderMode);			

		}

};



REGISTER_KERNEL_BUILDER(Name("RenderShadingSum").Device(DEVICE_GPU), RenderShadingSumOp)
