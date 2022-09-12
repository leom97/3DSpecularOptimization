#ifndef CONF
#define CONF

# include <string>

// Some constants: screen size, near and far plane
namespace conf {

	// Sensor configuration
	constexpr unsigned int SCR_WIDTH{ 300 };	// horizontal pixels of the sensor (so, it will be the horizontal pixels of the screen)
	constexpr unsigned int SCR_HEIGHT{ 400 };
	constexpr float near {.1f};	// near and far plane settings
	constexpr float far {15.0f};
	constexpr float focal{ near };	// focal length of camera (distance from sensor to camera origin)
	constexpr float fx{ focal / .0001f };	// usual fx = focal/size of a pixel, in camera units
	constexpr float fy{ focal / .0001f };	// usual fx = focal/size of a pixel, in camera units
	constexpr float cx{ SCR_WIDTH / 2 };	// x coordinate in pixels of the camera center, on the sensor
	constexpr float cy{ SCR_HEIGHT / 2 };	// y coordinate in pixels of the camera center, on the sensor
	constexpr float r{ focal / fx * (SCR_WIDTH - cx) };
	constexpr float l{ - focal / fx * cx };
	constexpr float t{ focal / fy * (SCR_HEIGHT - cy) };
	constexpr float b{ - focal / fy * cy };

	// Rendering configurations
	const std::string render_type = "color";	// normals, HDR, depth_map, otherwise it's the normal thing (color)
	const std::string depth_mode = "standard";	// one could go to other experimental modes, let's not do it

	// Shadow configuration
	//constexpr bool shadows{ false };
	//constexpr float scene_size{ 15.0f };  // i.e. we assume that the size is in ||x||<=scene_size
	//constexpr float light_nearPlane{ .1f };	// to render shadows from the perspective of light, we need a newar and far plane. This is the near. 
	//constexpr float light_farPlane{light_nearPlane + 2 * scene_size};	//The far is near + 2 scene_size

	// Light array
	// The object will be illuminated by some lights, arrange around a hemisphere, centered at the camera center, pointing towards the object
	// These lights can go up until the equator, or stop at an angle alpha
	constexpr float alpha{ 180.0f };	// aperture of the light array cone
	constexpr int N{ 3 };	// "number" of concentric rings in the light array (check MALTAB out to understand)

	// Output folder
	const std::string out_folder = "C:/Code/University/TUM/github/OpenGL/data/models/backpack/synthetic/non_lambertian/";

	// Input possibilities
	// If one already has generated a nice camera position, it is possible to start from that view, by setting the below option to true
	constexpr bool camera_from_file{ false };
	const std::string camera_path = "C:/Code/University/TUM/github/OpenGL/data/models/backpack/synthetic/non_lambertian/0_camera_pose_and_model.txt";	// fetch the camera pose from here (assuming it is a file generated from us)

	// Math constants
	constexpr float pi{ 3.14159265359f };

	// The geometric model, and the shaders
	const std::string model = "C:/Code/University/TUM/github/OpenGL/data/models/backpack/backpack.obj";
	const char * vertex_shader = "C:/Code/University/TUM/github/OpenGL/data/shaders/standard.vert";
	const char * fragment_shader = "C:/Code/University/TUM/github/OpenGL/data/shaders/standardMonoLight.frag";
	const char * shader_folder = "C:/Code/University/TUM/github/OpenGL/data/shaders/";

	// Other shaders
	const char* depth_to_screen_vert = "C:/Code/University/TUM/github/OpenGL/data/shaders/depthToScreenShader.vert";
	const char* depth_to_screen_frag = "C:/Code/University/TUM/github/OpenGL/data/shaders/depthToScreenShader.frag";
	const char* HDR_vert = "C:/Code/University/TUM/github/OpenGL/data/shaders/HDRToScreenShader.vert";
	const char* HDR_frag = "C:/Code/University/TUM/github/OpenGL/data/shaders/HDRToScreenShader.frag";
	const char* normals_vert = "C:/Code/University/TUM/github/OpenGL/data/shaders/normalsToScreenShader.vert";
	const char* normals_frag = "C:/Code/University/TUM/github/OpenGL/data/shaders/normalsToScreenShader.frag";
}

#endif