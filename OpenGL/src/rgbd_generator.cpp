#define STB_IMAGE_IMPLEMENTATION	// DO THIS ONLY IN ONE .cpp FILE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <shaderClass.h>
#include <camera.h> 
#include <model.h>

#include "conf.h"

//#include "filesystem.h"
#include <vector>
#include <fstream>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, Model& ourModel);
glm::vec3 getLightDir(float t, float f);
glm::mat4 getProjectionMatrix(float l, float r, float b, float t, float n, float f);
std::vector<double> lightArray(float alpha, int N);
glm::mat3 getRot(glm::mat4 T);


// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
bool lock{ true };  // the camera cannot move
float lastX = conf::SCR_WIDTH / 2.0f;
float lastY = conf::SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// light direction
float theta{ 0.0f }, phi{ 180.0f };   // in degrees
float v_deg{ 2.0f };    // how fast are those angles changing?

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// save snapshot
bool automatic_process{ false };    // it starts the automatic light saving
bool save{ false };
int nSnapshots{ 0 };

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(conf::SCR_WIDTH, conf::SCR_HEIGHT, "Window", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(true);

    // Use the reverse z trick
    if (conf::depth_mode == "reverse")    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    // build and compile shaders
    // -------------------------
    Shader normalShader(conf::vertex_shader, conf::fragment_shader);
    // load models
    // -----------
    Model ourModel(conf::model);
    ourModel.albedo_mode = 0;

    // lights
    // ------
    //glm::vec3 lightCol(10.0f, 10.0f, 10.0f);
    float lightIntensity{ 1.0f };
    normalShader.use();
    //normalShader.setVec3("light.color", lightCol);
    normalShader.setFloat("light.intensity", lightIntensity);
    normalShader.setInt("shadowMap", 2);
    std::vector<double> light_array = lightArray(conf::alpha, conf::N);   // light array in world frame
    int light_count{ -2 };   // says how many of the lights in the light array we have done. Start from -2 to also save albedos
    bool save_last_light = false;

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // camera loading
    if (conf::camera_from_file)
    {
        ifstream infile;
        infile.open(conf::camera_path);

        std::string line;
        for (int i{ 0 }; i < 18; i++) { getline(infile, line); }

        // Fetch position
        glm::vec3 Position;
        infile >> Position.x;
        infile >> Position.y;
        infile >> Position.z;
        // Fetch WorldUp
        glm::vec3 WorldUp;
        infile >> WorldUp.x;
        infile >> WorldUp.y;
        infile >> WorldUp.z;
        // Fetch yaw
        float Yaw;
        infile >> Yaw;
        // Fetch pitch
        float Pitch;
        infile >> Pitch;

        camera.Position = Position; camera.WorldUp = WorldUp; camera.Yaw = Yaw; camera.Pitch = Pitch;
        camera.updateCameraVectors();

    }
    else
    {
        //Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
    }

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window, ourModel);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // Matrices and other geometry
        //glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)conf::SCR_WIDTH / (float)conf::SCR_HEIGHT, conf::near, conf::far);
        glm::mat4 projection = getProjectionMatrix(conf::l, conf::r, conf::b, conf::t, conf::near, conf::far);
        if (conf::depth_mode == "reverse") {
            glm::mat4 maybe_Id = glm::mat4(1.0f);
            maybe_Id[2][2] = -0.5f;
            maybe_Id[3][2] = 0.5f;  // NB: 3rd column index, 2nd row index!
            projection = maybe_Id * projection;
        }
        glm::mat4 view = camera.GetViewMatrix();
        normalShader.setMat4("projection", projection);
        normalShader.setMat4("view", view);
        normalShader.setFloat("Near", conf::near);
        normalShader.setFloat("Far", conf::far);
        normalShader.setVec3("camPos", camera.Position);    // Camera positions for not just lambertian colors
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
        model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
        normalShader.setMat4("model", model);

        // For the light
        normalShader.use();
        glm::vec3 lightDir;
        if (!automatic_process)  lightDir = getLightDir(theta, phi);
        else
        {
            if (light_count == -2) {
                ourModel.only_HDR = true;   // salve only diffuse albedo
                ourModel.albedo_mode = 1;
            }
            if (light_count == -1) {
                ourModel.only_HDR = true;   // save only specular albedo
                ourModel.albedo_mode = 2;
            }
            if (light_count == 0)
            {
                ourModel.only_HDR = false;
                ourModel.albedo_mode = 0;

                std::string conf_path{ conf::out_folder + "conf.txt" };
                std::ofstream fout(conf_path);
                fout << std::setprecision(10);
                fout << "Number of frames\n" << light_array.size() / 3 << "\n";
                fout << "Height, width\n" << conf::SCR_HEIGHT << "\n" << conf::SCR_WIDTH << "\n";
                fout << "Depth mode\n" << conf::depth_mode << "\n";
                fout << "Near plane, far plane\n" << conf::near << "\n" << conf::far << "\n";
                fout << "Camera parameters: fx, fy, cx, cy\n" << conf::fx << "\n" << conf::fy << "\n" << conf::cx << "\n" << conf::cy << "\n";
                fout << "Light array paramters: alpha and N\n" << conf::alpha << "\n" << conf::N << "\n";
                fout << "HDR conventinon: color_HDR = the exact float color\n";
                fout << "Light convention: direction * intensity\n";
                fout << "Normal convention: Normal*0.5f+0.5f is outputted\n";

                fout.close();
            }
            else {
                ourModel.only_HDR = true;  // don't save again depth and normals...
            }

            save = true;

            // Rotate light (different from changing frame of reference)
            if (light_count >= 0) {
                lightDir.x = (float)light_array[3 * light_count];
                lightDir.y = (float)light_array[3 * light_count + 1];
                lightDir.z = (float)light_array[3 * light_count + 2];
                glm::mat3 R_WC = getRot(glm::inverse(view));
                lightDir = R_WC * lightDir; // Note: view = W -> C, but I should use C->W. Still in world frame
            }

            // Update counter
            ++light_count;
            if (light_count > 0)    std::cout << "///////////////////////// Light " << light_count << " of " << light_array.size() / 3 << std::endl;
            if (light_count == light_array.size() / 3) {
                automatic_process = false; light_count = 0; save_last_light = true;
            } // end at the end of the lights

        }
        normalShader.setVec3("light.wDir", lightDir);

        // Save stuff
        if (save) {
            save = false;
            lock = true;
            ourModel.save_to_txt = true;
            if (automatic_process) nSnapshots = light_count - 1;
            ourModel.nSnapshots = nSnapshots;

            if (light_count >= 1 || save_last_light)
            {
                // Save light
                std::string light_path{ conf::out_folder + std::to_string(nSnapshots) + "_" + "light_vector.txt" };
                std::ofstream fout(light_path);
                fout << std::setprecision(10);
                fout << "Saving light vector: intensity * direction. It is a vector in world coordinates, pointing towards the object.\n";
                std::vector<double> v = { lightIntensity * lightDir.x, lightIntensity * lightDir.y, lightIntensity * lightDir.z };
                std::copy(v.begin(), v.end(),
                    std::ostream_iterator<double>(fout, "\n"));
                std::cout << "Light vector successfully saved to " + light_path << std::endl;
                fout.close();
                save_last_light = false;
            }

            if (automatic_process && light_count == 1 || (!automatic_process && save_last_light))  // if I have set the first light, or we're not in automatic mode
            {
                // Save camera and model
                std::string camera_path{ conf::out_folder + std::to_string(nSnapshots) + "_" + "camera_pose_and_model.txt" };
                std::ofstream fout(camera_path);
                fout << std::setprecision(10);
                fout << "Camera pose (C->W). This is (i=0, j=0), (i=0, j=1)..., (i=1, j=0), (i=1, j=1)...\nThen Position, WorldUp, Yaw, Pitch\n";
                glm::mat4 cp = camera.GetViewMatrix();
                cp = glm::inverse(cp);  // C->W !

                std::vector<double> v;
                for (int i{ 0 }; i < 4; ++i) for (int j{ 0 }; j < 4; ++j)    v.push_back(cp[j][i]);
                std::copy(v.begin(), v.end(), std::ostream_iterator<double>(fout, "\n"));
                fout << camera.Position.x << "\n" << camera.Position.y << "\n" << camera.Position.z << "\n";
                fout << camera.WorldUp.x << "\n" << camera.WorldUp.y << "\n" << camera.WorldUp.z << "\n";
                fout << camera.Yaw << "\n";
                fout << camera.Pitch << "\n";

                fout << "Model matrix (M->W), saved in the same order as above\n";
                std::vector<double> w;
                for (int i{ 0 }; i < 4; ++i) for (int j{ 0 }; j < 4; ++j)    w.push_back(model[j][i]);
                std::copy(w.begin(), w.end(), std::ostream_iterator<double>(fout, "\n"));


                std::cout << "Camera pose and model matrix successfully saved to " + camera_path << std::endl;
                fout.close();
            }

            save = false;
            ++nSnapshots;
        }

        // Render the model
        ourModel.Draw(normalShader);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, Model& ourModel)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS && !lock)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !lock)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS && !lock)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS && !lock)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        save = true;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        theta += v_deg;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        theta -= v_deg;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        phi -= v_deg;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        phi += v_deg;
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        lock = false;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        lock = true;
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
        automatic_process = true;

}

glm::vec3 getLightDir(float t, float f)
{
    // t= theta in degrees, f = phi in degrees

    glm::mat4 ry = glm::mat4(1.0f);
    glm::mat4 rx = glm::mat4(1.0f);

    ry = glm::rotate(ry, glm::radians(f), glm::vec3(0.0f, 1.0f, 0.0f));
    rx = glm::rotate(rx, glm::radians(t), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 r = ry * rx;

    glm::vec3 res = glm::vec3(0.0f);
    res.x = r[2][0];
    res.y = r[2][1];
    res.z = r[2][2];

    return res;

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = (lock ? 0.0f : xpos - lastX);
    float yoffset = (lock ? 0.0f : lastY - ypos); // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (!lock) camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

glm::mat4 getProjectionMatrix(float l, float r, float b, float t, float n, float f) {
    // NB: l, b might as well be negative, n, f are positive!!

    float matrix[16];
    for (int i{ 0 }; i < 16; ++i) matrix[i] = 0.0f;

    matrix[0] = 2 * n / (r - l);
    matrix[5] = 2 * n / (t - b);
    matrix[8] = (r + l) / (r - l);
    matrix[9] = (t + b) / (t - b);
    matrix[10] = -(f + n) / (f - n);
    matrix[11] = -1;
    matrix[14] = -(2 * f * n) / (f - n);

    glm::mat4 res = glm::make_mat4(matrix);
    return res;
}

std::vector<double> lightArray(float alpha, int N)
{

    std::vector<double> points;
    alpha = alpha / 360 * 2 * conf::pi;
    float beta = alpha / (2 * N - 1);
    float l = beta;

    for (int i{ 1 }; i <= N; ++i)
    {
        float r_i = sin(beta / 2 + (i - 1) * beta);
        float z_i = -cos(beta / 2 + (i - 1) * beta);;
        int N_i = (int)(2 * conf::pi * r_i / l);
        for (int k{ 0 }; k <= N_i - 1; ++k)
        {
            float t_i = static_cast<float>(k) / N_i;
            points.push_back(r_i * sin(2 * conf::pi * t_i));
            points.push_back(r_i * cos(2 * conf::pi * t_i));
            points.push_back(z_i);
        }
    }

    return points;

}

// Gets 3x3 upper left block from a 4x4 matrix
glm::mat3 getRot(glm::mat4 T)
{
    glm::mat3 R;
    for (int i{ 0 }; i < 3; ++i)
    {
        for (int j{ 0 }; j < 3; ++j)
        {
            R[j][i] = T[j][i];
        }
    }
    return R;

}