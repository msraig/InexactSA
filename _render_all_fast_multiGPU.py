import tensorflow as tf
from tensorflow.python.framework import ops

renderer_all_fast_module = tf.load_op_library('./render_all_fast.so')
tf_renderer = renderer_all_fast_module.render_shading_sum

def fullRender(albedo_batch, spec_batch, roughness_batch, thetaPhi_batch, mask_batch,
					   lightID_batch, lightMat_batch,
					   matWorld, eyePosWorld, matView, matProj,
					   light_folder, light_file, num_sample_diffuse = 256, num_sample_spec = 1024, env_width = 256, 
					   instance_id = 'render_0', use_fresnel = 1, render_mode = 0):
	with tf.name_scope('FullRender_{}'.format(instance_id)):
		out_diffuse, out_spec = tf_renderer(
			albedo_batch,
			spec_batch,
			roughness_batch,
			thetaPhi_batch,
			mask_batch,
			lightID_batch,
			lightMat_batch,
			matWorld,
			eyePosWorld,
			matView,
			matProj,
			light_folder = light_folder,
			light_file = light_file,
			num_sample_diffuse = num_sample_diffuse,
			num_sample_spec = num_sample_spec,
			env_width = env_width,
			instance_id = instance_id,
			use_fresnel = use_fresnel,
			render_mode = render_mode
			)
		return out_diffuse / num_sample_diffuse + out_spec / num_sample_spec


def whiteBalanceRenderWithMask(albedo_batch, spec_batch, roughness_batch, thetaPhi_batch, mask_batch,
					   lightID_batch, lightMat_batch,
					   matWorld, eyePosWorld, matView, matProj,
					   light_folder, light_file, num_sample_diffuse = 256, num_sample_spec = 1024, env_width = 256, 
					   instance_id = 'render_0', use_fresnel = 1, render_mode = 0, wb = True, wb_scale = 0.5):
	with tf.name_scope('WB_{}'.format(instance_id)):
		wb_albedo = tf.ones_like(albedo_batch)
		
		tf_rendered_batch = fullRender(
			albedo_batch,
			spec_batch,
			roughness_batch,
			thetaPhi_batch,
			mask_batch,
			lightID_batch,
			lightMat_batch,
			matWorld,
			eyePosWorld,
			matView,
			matProj,
			light_folder = light_folder,
			light_file = light_file,
			num_sample_diffuse = num_sample_diffuse,
			num_sample_spec = num_sample_spec,
			env_width = env_width,
			instance_id = instance_id,
			use_fresnel = use_fresnel,
			render_mode = render_mode
			)	

		if(wb):
			tf_diffuse_batch = fullRender(
				wb_albedo,
				tf.zeros_like(spec_batch),
				0.5*tf.ones_like(roughness_batch),
				tf.zeros_like(thetaPhi_batch),
				mask_batch,
				lightID_batch,
				lightMat_batch,
				matWorld,
				eyePosWorld,
				matView,
				matProj,
				light_folder = light_folder,
				light_file = light_file,
				num_sample_diffuse = num_sample_diffuse,
				num_sample_spec = 1,
				env_width = env_width,
				instance_id = instance_id,
				use_fresnel = use_fresnel,
				render_mode = render_mode
				)

			wb_mean = tf.reduce_mean(tf_diffuse_batch, axis = [1,2,3], keep_dims = True)
			tf_rendered_batch_out = wb_scale * tf_rendered_batch / (wb_mean + 1e-8)
			return tf_rendered_batch_out
		else:
			return tf_rendered_batch



def whiteBalanceRender(albedo_batch, spec_batch, roughness_batch, thetaPhi_batch,
					   lightID_batch, lightMat_batch,
					   matWorld, eyePosWorld, matView, matProj,
					   light_folder, light_file, num_sample_diffuse = 256, num_sample_spec = 1024, env_width = 256, 
					   instance_id = 'render_0', use_fresnel = 1, render_mode = 0, wb = True, wb_scale = 0.5):
	with tf.name_scope('WB_{}'.format(instance_id)):
		mask = tf.ones_like(roughness_batch)
		wb_albedo = tf.ones_like(albedo_batch)
		
		tf_rendered_batch = fullRender(
			albedo_batch,
			spec_batch,
			roughness_batch,
			thetaPhi_batch,
			mask,
			lightID_batch,
			lightMat_batch,
			matWorld,
			eyePosWorld,
			matView,
			matProj,
			light_folder = light_folder,
			light_file = light_file,
			num_sample_diffuse = num_sample_diffuse,
			num_sample_spec = num_sample_spec,
			env_width = env_width,
			instance_id = instance_id,
			use_fresnel = use_fresnel,
			render_mode = render_mode
			)	

		if(wb):
			tf_diffuse_batch = fullRender(
				wb_albedo,
				tf.zeros_like(spec_batch),
				0.5*tf.ones_like(roughness_batch),
				tf.zeros_like(thetaPhi_batch),
				mask,
				lightID_batch,
				lightMat_batch,
				matWorld,
				eyePosWorld,
				matView,
				matProj,
				light_folder = light_folder,
				light_file = light_file,
				num_sample_diffuse = num_sample_diffuse,
				num_sample_spec = 1,
				env_width = env_width,
				instance_id = instance_id,
				use_fresnel = use_fresnel,
				render_mode = render_mode
				)

			wb_mean = tf.reduce_mean(tf_diffuse_batch, axis = [1,2,3], keep_dims = True)
			tf_rendered_batch_out = wb_scale * tf_rendered_batch / (wb_mean + 1e-8)
			return tf_rendered_batch_out
		else:
			return tf_rendered_batch


		


	
