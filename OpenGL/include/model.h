#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <mesh.h>
#include <shaderClass.h>

#include <string>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "conf.h"

using namespace std;

unsigned int TextureFromFile(const char *path, const string &directory, bool gamma = false);

class Model 
{
public:
    // model data 
    vector<Texture> textures_loaded;	// stores all the textures loaded so far, optimization to make sure textures aren't loaded more than once.
    vector<Mesh>    meshes;
    string directory;
    bool gammaCorrection;

    // plane on which to render framebuffer textures
    unsigned int plVAO;
    unsigned int plVBO;

    // depth map data
    Shader depthToScreenShader = Shader(conf::depth_to_screen_vert, conf::depth_to_screen_frag);
    unsigned int depthMapFBO;  
    unsigned int depthMap;  // this is a texture, perhaps rename it      

    // HDR data
    Shader HDRToScreenShader = Shader(conf::HDR_vert,conf::HDR_frag);
    unsigned int HDRFBO;
    unsigned int HDRTex;

    // normals data
    Shader normalsToScreenShader = Shader(conf::normals_vert, conf::normals_frag);
    unsigned int normalsFBO;
    unsigned int normalsTex;

    // save to files
    bool save_to_txt{ false };
    int nSnapshots{ 0 };
    bool only_HDR{ false };
    int albedo_mode{ 0 };   // 0: render normal HDR, 1: render diffuse albedo, 2: render specular albedo

    // constructor, expects a filepath to a 3D model.
    Model(string const& path, bool gamma = false) : gammaCorrection(gamma)
    {

        // Load the model
        loadModel(path);

        // The all-purpose rendering "quad", on which to render framebuffers content
        createPlaneObject();

        // Create various framebuffers
        createDepthFB(); 
        createHDRFB();
        createnormalsFB();

        // Configure x-toScreenShader s
        if (conf::depth_mode == "reverse") 
        { 
            depthToScreenShader.use();
            depthToScreenShader.setInt("reverse", 1);  // Set a different conversion back to the eye space depth
            glDepthFunc(GL_GREATER); // https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
        }     
        else
        {
            depthToScreenShader.use();
            depthToScreenShader.setInt("reverse", 0);
        }
        HDRToScreenShader.use();
        HDRToScreenShader.setInt("HDRTexture", 0);
        normalsToScreenShader.use();
        normalsToScreenShader.setInt("normalsTexture", 0);

        glEnable(GL_DEPTH_TEST);

    }

    // draws the model, and thus all its meshes (to different framebuffers)
    void Draw(Shader &normalShader)
    {
        normalShader.setInt("albedo_mode", albedo_mode);
        // Save depth map to appropriate framebuffer
        // Save HDR color information to appropriate framebuffer
        scene_to_FB(normalShader, depthMapFBO);
        scene_to_FB(normalShader, HDRFBO);
        scene_to_FB(normalShader, normalsFBO, true);    // special modification in case of normals, to the default shaders

        // Save depth to file
        if (save_to_txt)
        {
            if (!only_HDR) depthMapToFile(depthMap, conf::out_folder + std::to_string(nSnapshots) + "_" + "depth_map_" + conf::depth_mode + ".txt");
            if (albedo_mode == 0)
                HDRTexToFile(HDRTex, conf::out_folder + std::to_string(nSnapshots) + "_" + "HDRAlpha.txt");
            else if (albedo_mode == 1)
                HDRTexToFile(HDRTex, conf::out_folder + std::to_string(nSnapshots+2) + "_" + "diffuse_albedo.txt");
            else if (albedo_mode ==  2)
                HDRTexToFile(HDRTex, conf::out_folder + std::to_string(nSnapshots+1) + "_" + "specular_albedo.txt");
            if (!only_HDR) normalsTexToFile(normalsFBO, conf::out_folder + std::to_string(nSnapshots) + "_" + "normals.txt");
            save_to_txt = false;
        }

        // Print something to screen
        if (conf::render_type == "depth_map") view_depth_FBO(); // visualize the depth map to screen
        else if (conf::render_type == "HDR")    view_HDR_FBO(); // visualize HDR texture to screen
        else if (conf::render_type == "normals")    view_normals_FBO(); // visualize normals
        else scene_to_FB(normalShader, 0); // the RGB image to screen
    }
    
private:

    // clear stuff
    void clear_buffers()
    {
        if (conf::depth_mode == "reverse") { glClearDepth(0.0f); }      // https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // Render the scene to the specified framebuffer, using the specified shader
    void scene_to_FB(Shader shader, unsigned int FBId, bool normals = false)
    {
        if (normals)
        {
            shader.use();
            shader.setInt("normal_map", 1);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, FBId);
        clear_buffers();
        render_scene(shader); // note, using the same shader as for the color scene. This doesn't matter as the fragment shader will just output gl_FragCoord.z
        
        shader.use();
        shader.setInt("normal_map", 0);
    }

    // Applies the rendering function to all the meshes, called from scene_to_FB
    void render_scene(Shader shader) 
    {
        for (unsigned int i = 0; i < meshes.size(); i++) meshes[i].Draw(shader);
    }

    // render depth map on default framebuffer
    void view_depth_FBO()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (conf::depth_mode == "reverse") { glDepthFunc(GL_LESS); }     // Do I have to keep this? https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the depth map
        depthToScreenShader.use();
        depthToScreenShader.setFloat("near_plane", conf::near);
        depthToScreenShader.setFloat("far_plane", conf::far);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthMap);	// take the depth map texture, bind it as the current 0-th texture (which is okay, since we have set the uniform sampler to 0 in the screen shaders, see createDepthFB)
        glBindVertexArray(plVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glEnable(GL_DEPTH_TEST); // re-enable for future renderings in the first pass
        if (conf::depth_mode == "reverse") { glDepthFunc(GL_GREATER); }     // https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/

    }

    // render HDR color to screen
    void view_HDR_FBO()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw HDR texture
        HDRToScreenShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, HDRTex);
        glBindVertexArray(plVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glEnable(GL_DEPTH_TEST); // re-enable for future renderings in the first pass
        if (conf::depth_mode == "reverse") { glDepthFunc(GL_GREATER); }     // https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/

    }

    // render normals map to screen
    void view_normals_FBO()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw HDR texture
        normalsToScreenShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, normalsTex);
        glBindVertexArray(plVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glEnable(GL_DEPTH_TEST); // re-enable for future renderings in the first pass
        if (conf::depth_mode == "reverse") { glDepthFunc(GL_GREATER); }     // https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/

    }

    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
    void loadModel(string const &path)
    {
        // read file via ASSIMP
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        // check for errors
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
            return;
        }
        // retrieve the directory path of the filepath
        directory = path.substr(0, path.find_last_of('/'));

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene);
    }

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
    void processNode(aiNode *node, const aiScene *scene)
    {
        // process each mesh located at the current node
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for(unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }

    }

    Mesh processMesh(aiMesh *mesh, const aiScene *scene)
    {
        // data to fill
        vector<Vertex> vertices;
        vector<unsigned int> indices;
        vector<Texture> textures;

        // walk through each of the mesh's vertices
        for(unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
            // positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.Position = vector;
            // normals
            if (mesh->HasNormals())
            {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if(mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                glm::vec2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.x = mesh->mTextureCoords[0][i].x; 
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;
                // tangent
                vector.x = mesh->mTangents[i].x;
                vector.y = mesh->mTangents[i].y;
                vector.z = mesh->mTangents[i].z;
                vertex.Tangent = vector;
                // bitangent
                vector.x = mesh->mBitangents[i].x;
                vector.y = mesh->mBitangents[i].y;
                vector.z = mesh->mBitangents[i].z;
                vertex.Bitangent = vector;
            }
            else
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);

            vertices.push_back(vertex);
        }
        // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for(unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            // retrieve all indices of the face and store them in the indices vector
            for(unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);        
        }
        // process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];    
        // we assume a convention for sampler names in the shaders. Each diffuse texture should be named
        // as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
        // Same applies to other texture as the following list summarizes:
        // diffuse: texture_diffuseN
        // specular: texture_specularN
        // normal: texture_normalN

        // 1. diffuse maps
        vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        // 2. specular maps
        vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        // 3. normal maps
        std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
        // 4. height maps
        std::vector<Texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
        textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());
        
        // return a mesh object created from the extracted mesh data
        return Mesh(vertices, indices, textures);
    }

    // checks all material textures of a given type and loads the textures if they're not loaded yet.
    // the required info is returned as a Texture struct.
    vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName)
    {
        vector<Texture> textures;
        for(unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
            bool skip = false;
            for(unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if(std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                    break;
                }
            }
            if(!skip)
            {   // if texture hasn't been loaded already, load it
                Texture texture;
                texture.id = TextureFromFile(str.C_Str(), this->directory);
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);  // store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
            }
        }
        return textures;
    }

    // Create a framebuffer for the depth map if necessary
    void createDepthFB() {
        // the framebuffer
        glGenFramebuffers(1, &depthMapFBO);
        // completion: attacching a texture
        glGenTextures(1, &depthMap);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, conf::SCR_WIDTH, conf::SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);  // i.e. allocate memory, to be filled later at rendering
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, conf::SCR_WIDTH, conf::SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);  // i.e. allocate memory, to be filled later at rendering

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // bind framebuffer and attach depth texture
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
        glDrawBuffer(GL_NONE);  // i.e. explicitly tell we want no color data
        glReadBuffer(GL_NONE);  // i.e. explicitly tell we want no color data

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cout << "Failed to create depth framebuffer." << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        depthToScreenShader.use();
        depthToScreenShader.setInt("depthMap", 0);
    }

    // Create floating point color framebuffer to store HDR values
    void createHDRFB()
    {
        // framebuffer
        glGenFramebuffers(1, &HDRFBO);

        // completion
        glGenTextures(1, &HDRTex);
        glBindTexture(GL_TEXTURE_2D, HDRTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, conf::SCR_WIDTH, conf::SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // must also create a depth renderbuffer (sure?)
        unsigned int rboDepth;
        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, conf::SCR_WIDTH, conf::SCR_HEIGHT);
        // attach buffers
        glBindFramebuffer(GL_FRAMEBUFFER, HDRFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, HDRTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

    }

    //  Framebuffer for normal map (floating point too, out of laziness)
    void createnormalsFB()
    {
        // framebuffer
        glGenFramebuffers(1, &normalsFBO);

        // completion
        glGenTextures(1, &normalsTex);
        glBindTexture(GL_TEXTURE_2D, normalsTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, conf::SCR_WIDTH, conf::SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // must also create a depth renderbuffer (sure?)
        unsigned int rboDepth;
        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, conf::SCR_WIDTH, conf::SCR_HEIGHT);
        // attach buffers
        glBindFramebuffer(GL_FRAMEBUFFER, normalsFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalsTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Create a rectangle which covers the screen, to which render depth maps
    void createPlaneObject()
    {
        float vertices[] = {
            // positions         // texCoords
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,

            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,  1.0f, 1.0f
        };


        glGenVertexArrays(1, &plVAO);
        glBindVertexArray(plVAO);

        glGenBuffers(1, &plVBO);
        glBindBuffer(GL_ARRAY_BUFFER, plVBO);

        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    }

    // Saves the depth map texture stored in the texture depthMap, to .txt file, to avoid loosing precision
    void depthMapToFile(unsigned int depthMap, std::string path)
    {

        depthToScreenShader.use();
        depthToScreenShader.setFloat("near_plane", conf::near);
        depthToScreenShader.setFloat("far_plane", conf::far);

        GLfloat* d = new GLfloat[conf::SCR_WIDTH * conf::SCR_HEIGHT];
        //glReadPixels(0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);
        //glReadPixels(0, 0, conf::SCR_WIDTH, conf::SCR_HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);

        glBindTexture(GL_TEXTURE_2D, depthMap);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, d);

        std::vector<double> v(d, d + static_cast<int>(conf::SCR_WIDTH * conf::SCR_HEIGHT));

        std::ofstream fout(path);
        fout << std::setprecision(10);

        std::copy(v.begin(), v.end(),
            std::ostream_iterator<double>(fout, "\n"));

        fout.close();
        delete[] d;

        std::cout << "Depth map successfully saved to " + path << std::endl;
    }

    // Saves the HDR RGB data to file, be able to recover then the true intensity
    void HDRTexToFile(unsigned int HDRTex, std::string path)
    {

        GLfloat* c = new GLfloat[conf::SCR_WIDTH * conf::SCR_HEIGHT * 4];
        //glReadPixels(0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);
        //glReadPixels(0, 0, conf::SCR_WIDTH, conf::SCR_HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);

        glBindTexture(GL_TEXTURE_2D, HDRTex);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, c);

        std::vector<double> v(c, c + static_cast<int>(conf::SCR_WIDTH * conf::SCR_HEIGHT * 4));

        std::ofstream fout(path);
        fout << std::setprecision(10);

        std::copy(v.begin(), v.end(),
            std::ostream_iterator<double>(fout, "\n"));

        fout.close();
        delete[] c;

        std::cout << "HDR color successfully saved to " + path << std::endl;
    }

    // Saves normal map to txt file (n -> n/2 + 1/2 -> txt file)
    void normalsTexToFile(unsigned int HDRTex, std::string path)
    {
        GLfloat* c = new GLfloat[conf::SCR_WIDTH * conf::SCR_HEIGHT * 3];
        //glReadPixels(0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);
        //glReadPixels(0, 0, conf::SCR_WIDTH, conf::SCR_HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, &d[0]);

        glBindTexture(GL_TEXTURE_2D, normalsTex);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, c);

        std::vector<double> v(c, c + static_cast<int>(conf::SCR_WIDTH * conf::SCR_HEIGHT * 3));

        std::ofstream fout(path);
        fout << std::setprecision(10);

        std::copy(v.begin(), v.end(),
            std::ostream_iterator<double>(fout, "\n"));

        fout.close();
        delete[] c;

        std::cout << "Normals successfully saved to " + path << std::endl;
    }

};

unsigned int TextureFromFile(const char *path, const string &directory, bool gamma)
{
    string filename = string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

#endif