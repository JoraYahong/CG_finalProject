#include <glad/glad.h>
#include <GLFW/glfw3.h>
// FreeType
#include <ft2build.h>
#include <freetype/freetype.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <../shader.h>
#include <../camera.h>
#include <../model.h>

#include "utils.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <unordered_map>
#include <../shader_character.h>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
unsigned int loadCubemap(std::string faces[]);
void RenderText(Shader &shader, std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color);


GLboolean CheckCollision(float x_max1, float y_max1, float z_max1,
	float x_min1, float y_min1, float z_min1,
	float x_max2, float y_max2, float z_max2,
	float x_min2, float y_min2, float z_min2,
	float x1_pos, float y1_pos, float z1_pos,
	float x2_pos, float y2_pos, float z2_pos);
// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
bool firstMouse = true;




// timing
float deltaTime = 0.0f;
float lastFrame = (float)glfwGetTime();
// lighting
glm::vec3 lightPos(1.0f, -0.35f, 2.0f);

float rotate_step = 0.0f;
float translat_step = 0.05f;
float translat_step2 = 0.0f;
float run_speed = 1.0f;
float current_speed = 0.0f;
float current_rotate_speed = 0.0f;
float rotate_speed = 200.0f;
float camera_zoomin = 75.0f;
float dx = 0.0f;
float dz= 0.0f;
float dy = 0.0f;
float x_position = 0.35f;
float y_position = -0.35f;
float z_position = 2.0f;
float gravity = -12.0f;
float jump_power = 4.0f;
float jump_upward_speed = 0.0f;
bool isOnSky = false;
float terrain_height = -0.35f;
float cameraHeight = 0.3f;
float barrel_r = 0.0f;
float barrel_p = 0.0f;
int score = 0;
int score1 = 0;
int score2 = score1;
// camera
Camera camera1(true,rotate_step,x_position, y_position, z_position,glm::vec3(0.0f, 0.0f, 0.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
Camera camera2(false,rotate_step, x_position-0.3f, cameraHeight, z_position, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 3.0f, 0.0f),90.0f,-90.0f);


struct Character {
	GLuint TextureID;   // ID handle of the glyph texture
	glm::ivec2 Size;    // Size of glyph
	glm::ivec2 Bearing;  // Offset from baseline to left/top of glyph
	GLuint Advance;    // Horizontal offset to advance to next glyph
};

std::map<GLchar, Character> Characters;
GLuint VAO, VBO;
struct VertexA {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec4 boneIds = glm::vec4(0);
	glm::vec4 boneWeights = glm::vec4(0.0f);
	
};

struct box {
	float max_x = 0.0f;
	float max_y = 0.0f;
	float max_z = 0.0f;
	float min_x = 10000.0f;
	float min_y = 10000.0f;
	float min_z = 10000.0f;
};

struct Bone {
	int id = 0; // position of the bone in final upload array
	std::string name = "";
	glm::mat4 offset = glm::mat4(1.0f);
	std::vector<Bone> children = {};
};

struct BoneTransformTrack {
	std::vector<float> positionTimestamps = {};
	std::vector<float> rotationTimestamps = {};
	std::vector<float> scaleTimestamps = {};

	std::vector<glm::vec3> positions = {};
	std::vector<glm::quat> rotations = {};
	std::vector<glm::vec3> scales = {};
};

struct Animation {
	float duration = 0.0f;
	float ticksPerSecond = 1.0f;
	std::unordered_map<std::string, BoneTransformTrack> boneTransforms = {};
};

bool readSkeleton(Bone& boneOutput, aiNode* node, std::unordered_map<std::string, std::pair<int, glm::mat4>>& boneInfoTable) {

	if (boneInfoTable.find(node->mName.C_Str()) != boneInfoTable.end()) { // if node is actually a bone
		boneOutput.name = node->mName.C_Str();
		boneOutput.id = boneInfoTable[boneOutput.name].first;
		boneOutput.offset = boneInfoTable[boneOutput.name].second;

		for (int i = 0; i < node->mNumChildren; i++) {
			Bone child;
			readSkeleton(child, node->mChildren[i], boneInfoTable);
			boneOutput.children.push_back(child);
		}
		return true;
	}
	else { // find bones in children
		for (int i = 0; i < node->mNumChildren; i++) {
			if (readSkeleton(boneOutput, node->mChildren[i], boneInfoTable)) {
				return true;
			}

		}
	}
	return false;
}

box loadModel(const aiScene* scene, aiMesh* mesh, std::vector<VertexA>& verticesOutput, std::vector<uint>& indicesOutput, Bone& skeletonOutput, uint &nBoneCount) {
	box box1;
	verticesOutput = {};
	indicesOutput = {};
	//load position, normal, uv
	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
		//process position 
		VertexA vertexA;
		glm::vec3 vector;
		vector.x = mesh->mVertices[i].x;
		vector.y = mesh->mVertices[i].y;
		vector.z = mesh->mVertices[i].z;
		vertexA.position = vector;
		if (glm::abs(vector.x) > box1.max_x) box1.max_x = vector.x;
		if (glm::abs(vector.y) > box1.max_y) box1.max_y = vector.y;
		if (glm::abs(vector.z) > box1.max_z) box1.max_z = vector.z;
		if (glm::abs(vector.x) < box1.min_x) box1.min_x = vector.x;
		if (glm::abs(vector.y) < box1.min_y) box1.min_y = vector.y;
		if (glm::abs(vector.z) < box1.min_z) box1.min_z = vector.z;
		//cout << box1.max_x << box1.min_z << std::endl;
		//process normal
		vector.x = mesh->mNormals[i].x;
		vector.y = mesh->mNormals[i].y;
		vector.z = mesh->mNormals[i].z;
		vertexA.normal = vector;
		//process uv
		glm::vec2 vec;
		vec.x = mesh->mTextureCoords[0][i].x;
		vec.y = mesh->mTextureCoords[0][i].y;
		vertexA.uv = vec;

		vertexA.boneIds = glm::ivec4(0);
		vertexA.boneWeights = glm::vec4(0.0f);

		verticesOutput.push_back(vertexA);
	}

	//load boneData to vertices
	std::unordered_map<std::string, std::pair<int, glm::mat4>> boneInfo = {};
	std::vector<uint> boneCounts;
	boneCounts.resize(verticesOutput.size(), 0);
	nBoneCount = mesh->mNumBones;

	//loop through each bone
	for (uint i = 0; i < nBoneCount; i++) {
		aiBone* bone = mesh->mBones[i];
		glm::mat4 m = assimpToGlmMatrix(bone->mOffsetMatrix);
		boneInfo[bone->mName.C_Str()] = { i, m };

		//loop through each VertexA that have that bone
		for (int j = 0; j < bone->mNumWeights; j++) {
			uint id = bone->mWeights[j].mVertexId;
			float weight = bone->mWeights[j].mWeight;
			boneCounts[id]++;
			switch (boneCounts[id]) {
			case 1:
				verticesOutput[id].boneIds.x = i;
				verticesOutput[id].boneWeights.x = weight;
				break;
			case 2:
				verticesOutput[id].boneIds.y = i;
				verticesOutput[id].boneWeights.y = weight;
				break;
			case 3:
				verticesOutput[id].boneIds.z = i;
				verticesOutput[id].boneWeights.z = weight;
				break;
			case 4:
				verticesOutput[id].boneIds.w = i;
				verticesOutput[id].boneWeights.w = weight;
				break;
			default:
				//std::cout << "err: unable to allocate bone to VertexA" << std::endl;
				break;

			}
		}
	}



	//normalize weights to make all weights sum 1
	for (int i = 0; i < verticesOutput.size(); i++) {
		glm::vec4 & boneWeights = verticesOutput[i].boneWeights;
		float totalWeight = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
		if (totalWeight > 0.0f) {
			verticesOutput[i].boneWeights = glm::vec4(
				boneWeights.x / totalWeight,
				boneWeights.y / totalWeight,
				boneWeights.z / totalWeight,
				boneWeights.w / totalWeight
			);
		}
	}


	//load indices
	for (int i = 0; i < mesh->mNumFaces; i++) {
		aiFace& face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++)
			indicesOutput.push_back(face.mIndices[j]);
	}

	// create bone hirerchy
	readSkeleton(skeletonOutput, scene->mRootNode, boneInfo);
	return box1;
}

void loadAnimation(const aiScene* scene, Animation& animation) {
	//loading  first Animation
	aiAnimation* anim = scene->mAnimations[0];

	if (anim->mTicksPerSecond != 0.0f)
		animation.ticksPerSecond = anim->mTicksPerSecond;
	else
		animation.ticksPerSecond = 1;


	animation.duration = anim->mDuration * anim->mTicksPerSecond;
	animation.boneTransforms = {};

	//load positions rotations and scales for each bone
	// each channel represents each bone
	for (int i = 0; i < anim->mNumChannels; i++) {
		aiNodeAnim* channel = anim->mChannels[i];
		BoneTransformTrack track;
		for (int j = 0; j < channel->mNumPositionKeys; j++) {
			track.positionTimestamps.push_back(channel->mPositionKeys[j].mTime);
			track.positions.push_back(assimpToGlmVec3(channel->mPositionKeys[j].mValue));
		}
		for (int j = 0; j < channel->mNumRotationKeys; j++) {
			track.rotationTimestamps.push_back(channel->mRotationKeys[j].mTime);
			track.rotations.push_back(assimpToGlmQuat(channel->mRotationKeys[j].mValue));

		}
		for (int j = 0; j < channel->mNumScalingKeys; j++) {
			track.scaleTimestamps.push_back(channel->mScalingKeys[j].mTime);
			track.scales.push_back(assimpToGlmVec3(channel->mScalingKeys[j].mValue));

		}
		animation.boneTransforms[channel->mNodeName.C_Str()] = track;
	}
}

unsigned int createVertexArray(std::vector<VertexA>& vertices, std::vector<uint> indices) {
	uint
		vao = 0,
		vbo = 0,
		ebo = 0;

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexA) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexA), (GLvoid*)offsetof(VertexA, position));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexA), (GLvoid*)offsetof(VertexA, normal));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexA), (GLvoid*)offsetof(VertexA, uv));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexA), (GLvoid*)offsetof(VertexA, boneIds));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(VertexA), (GLvoid*)offsetof(VertexA, boneWeights));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), &indices[0], GL_STATIC_DRAW);
	glBindVertexArray(0);
	return vao;
}

uint createTexture(std::string filepath) {
	uint textureId = 0;
	int width, height, nrChannels;
	byte* data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 4);
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 3);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

	stbi_image_free(data);
	glBindTexture(GL_TEXTURE_2D, 0);
	return textureId;
}



std::pair<uint, float> getTimeFraction(std::vector<float>& times, float& dt) {
	uint segment = 0;
	while (dt > times[segment]) {
		segment++;
	}
	float start = times[segment - 1];
	float end = times[segment];
	float frac = (dt - start) / (end - start);
	return { segment, frac };
}



void getPose(Animation& animation, Bone& skeletion, float dt, std::vector<glm::mat4>& output, glm::mat4 &parentTransform, glm::mat4& globalInverseTransform) {
	BoneTransformTrack& btt = animation.boneTransforms[skeletion.name];
	dt = fmod(dt, animation.duration);
	std::pair<uint, float> fp;
	//calculate interpolated position
	fp = getTimeFraction(btt.positionTimestamps, dt);

	glm::vec3 position1 = btt.positions[fp.first - 1];
	glm::vec3 position2 = btt.positions[fp.first];

	glm::vec3 position = glm::mix(position1, position2, fp.second);

	//calculate interpolated rotation
	fp = getTimeFraction(btt.rotationTimestamps, dt);
	glm::quat rotation1 = btt.rotations[fp.first - 1];
	glm::quat rotation2 = btt.rotations[fp.first];

	glm::quat rotation = glm::slerp(rotation1, rotation2, fp.second);

	//calculate interpolated scale
	fp = getTimeFraction(btt.scaleTimestamps, dt);
	glm::vec3 scale1 = btt.scales[fp.first - 1];
	glm::vec3 scale2 = btt.scales[fp.first];

	glm::vec3 scale = glm::mix(scale1, scale2, fp.second);

	glm::mat4 positionMat = glm::mat4(1.0),
		scaleMat = glm::mat4(1.0);


	// calculate localTransform
	positionMat = glm::translate(positionMat, position);
	glm::mat4 rotationMat = glm::toMat4(rotation);
	scaleMat = glm::scale(scaleMat, scale);
	glm::mat4 localTransform = positionMat * rotationMat * scaleMat;
	glm::mat4 globalTransform = parentTransform * localTransform;

	output[skeletion.id] = globalInverseTransform * globalTransform * skeletion.offset;
	//update values for children bones
	for (Bone& child : skeletion.children) {
		getPose(animation, child, dt, output, globalTransform, globalInverseTransform);
	}
	//std::cout << dt << " => " << position.x << ":" << position.y << ":" << position.z << ":" << std::endl;
}







int main()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	float SIZE = 5.0f;

	float SKYVERTICES[] = {
		-SIZE,  SIZE, -SIZE,
		-SIZE, -SIZE, -SIZE,
		SIZE, -SIZE, -SIZE,
		 SIZE, -SIZE, -SIZE,
		 SIZE,  SIZE, -SIZE,
		-SIZE,  SIZE, -SIZE,

		-SIZE, -SIZE,  SIZE,
		-SIZE, -SIZE, -SIZE,
		-SIZE,  SIZE, -SIZE,
		-SIZE,  SIZE, -SIZE,
		-SIZE,  SIZE,  SIZE,
		-SIZE, -SIZE,  SIZE,

		 SIZE, -SIZE, -SIZE,
		 SIZE, -SIZE,  SIZE,
		 SIZE,  SIZE,  SIZE,
		 SIZE,  SIZE,  SIZE,
		 SIZE,  SIZE, -SIZE,
		 SIZE, -SIZE, -SIZE,

		-SIZE, -SIZE,  SIZE,
		-SIZE,  SIZE,  SIZE,
		 SIZE,  SIZE,  SIZE,
		 SIZE,  SIZE,  SIZE,
		 SIZE, -SIZE,  SIZE,
		-SIZE, -SIZE,  SIZE,

		-SIZE,  SIZE, -SIZE,
		 SIZE,  SIZE, -SIZE,
		 SIZE,  SIZE,  SIZE,
		 SIZE,  SIZE,  SIZE,
		-SIZE,  SIZE,  SIZE,
		-SIZE,  SIZE, -SIZE,

		-SIZE, -SIZE, -SIZE,
		-SIZE, -SIZE,  SIZE,
		 SIZE, -SIZE, -SIZE,
		 SIZE, -SIZE, -SIZE,
		-SIZE, -SIZE,  SIZE,
		 SIZE, -SIZE,  SIZE
	};
	std::string faces[] =
	{
		"../Project2/resources/skybox/right.jpg",
		"../Project2/resources/skybox/left.jpg",
		"../Project2/resources/skybox/top.jpg",
		"../Project2/resources/skybox/bottom.jpg",
		"../Project2/resources/skybox/back.jpg",
		"../Project2/resources/skybox/front.jpg"
	};
	
	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "MyGame", NULL, NULL);
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

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);
	// Set OpenGL options
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//load model file
	Assimp::Importer importer;
	const char* filePath = "../Project2/resources/man/model.dae";
	const aiScene* scene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
	}
	aiMesh* mesh = scene->mMeshes[0];

	std::vector<VertexA> vertices = {};
	std::vector<uint> indices = {};
	uint boneCount = 0;
	Animation animation;
	uint vao = 0;
	Bone skeleton;
	uint diffuseTexture;

	//inverse the global transform
	glm::mat4 globalInverseTransform = assimpToGlmMatrix(scene->mRootNode->mTransformation);
	globalInverseTransform = glm::inverse(globalInverseTransform);

	box box_man = loadModel(scene, mesh, vertices, indices, skeleton, boneCount);
	loadAnimation(scene, animation);

	vao = createVertexArray(vertices, indices);
	diffuseTexture = createTexture("../Project2/resources/man//diffuse.png");


	glm::mat4 identity(1.0);

	std::vector<glm::mat4> currentPose = {};
	currentPose.resize(boneCount, identity); 

	uint shader = createShader(vertexShaderSource, fragmentShaderSource);
	//get all shader uniform locations
	uint viewProjectionMatrixLocation = glGetUniformLocation(shader, "view_projection_matrix");
	uint modelMatrixLocation = glGetUniformLocation(shader, "model_matrix");
	uint boneMatricesLocation = glGetUniformLocation(shader, "bone_transforms");
	uint textureLocation = glGetUniformLocation(shader, "diff_texture");


	Shader textShader("../Project2/textShader.vs", "../Project2/textShader.fs");
	glm::mat4 projection_text = glm::ortho(0.0f, static_cast<GLfloat>(1200), 0.0f, static_cast<GLfloat>(900));
	textShader.use();
	glUniformMatrix4fv(glGetUniformLocation(textShader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection_text));

	Shader ourShader("../Project2/1.model_loading.vs", "../Project2/1.model_loading.fs");
	// load models
	// -----------
	Model ourModel("../Project2/resources/snake-ground/source/snake.obj");
	Model lantern("../Project2/resources/lantern/lantern.obj");
	Model barrel("../Project2/resources/barrel/barrel.obj");
	unsigned int specularMap = loadTexture("../Project2/resources/lantern/lanternS.png");
	unsigned int diffuseMap = loadTexture("../Project2/resources/lantern/lantern.png");

	unsigned int specularMap_barrel = loadTexture("../Project2/resources/barrel/barrelS.png");
	unsigned int diffuseMap_barrel = loadTexture("../Project2/resources/barrel/barrel1.png");
	//load Shader
	Shader skyboxShader("../Project2/skybox.vs", "../Project2/skybox.fs");
	Shader lightingShader("../Project2/lantern.vs", "../Project2/lantern.fs");
	lightingShader.use();
	lightingShader.setInt("material.diffuse", 0);
	lightingShader.setInt("material.specular", 1);


	// FreeType
	FT_Library ft;
	// All functions return a value different than 0 whenever an error occurred
	if (FT_Init_FreeType(&ft))
		std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;

	// Load font as face
	FT_Face face;
	if (FT_New_Face(ft, "../Project2/resources/arial.ttf", 0, &face))
		cout << "ERROR::FREETYPE: Failed to load font" << endl;

	// Set size to load glyphs as
	FT_Set_Pixel_Sizes(face, 0, 48);

	// Disable byte-alignment restriction
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Load first 128 characters of ASCII set
	for (GLubyte c = 0; c < 128; c++)
	{
		// Load character glyph 
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
			continue;
		}
		// Generate texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RED,
			face->glyph->bitmap.width,
			face->glyph->bitmap.rows,
			0,
			GL_RED,
			GL_UNSIGNED_BYTE,
			face->glyph->bitmap.buffer
		);
		// Set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Now store character for later use
		Character character = {
			texture,
			glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
			face->glyph->advance.x
		};
		Characters.insert(std::pair<GLchar, Character>(c, character));
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	// Destroy FreeType once we're finished
	FT_Done_Face(face);
	FT_Done_FreeType(ft);


	// Configure VAO/VBO for texture quads
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//skyboxVAO
	unsigned int skyboxVAO, skyboxVBO;
	glGenVertexArrays(1, &skyboxVAO);
	glGenBuffers(1, &skyboxVBO);
	glBindVertexArray(skyboxVAO);
	glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SKYVERTICES), &SKYVERTICES, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	unsigned int cubemapTexture = loadCubemap(faces);
	skyboxShader.use();
	skyboxShader.setInt("skybox", 0);


	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		
		
		
		float elapsedTime = (float)glfwGetTime();
		deltaTime = elapsedTime - lastFrame;
		float dAngle = elapsedTime * 0.002;
		lastFrame = elapsedTime;
		processInput(window);
		rotate_step += current_rotate_speed * deltaTime;
		float distance = current_speed * deltaTime;
		dx = -(float)(distance * glm::sin(glm::radians(rotate_step)));
		dz = (float)(distance * glm::cos(glm::radians(rotate_step)));
		x_position += dx;
		z_position += dz;
		barrel_r += deltaTime * 200.0f;
		jump_upward_speed += gravity * deltaTime;
		dy = jump_upward_speed * deltaTime;
		y_position += dy;
		if (y_position < terrain_height) {
			jump_upward_speed = 0.0f;
			y_position = terrain_height;
			isOnSky = false;
		}
		camera1.changePosition(rotate_step, x_position, y_position, z_position);
		barrel_p += deltaTime * 1.5f;
		if (barrel_p > 22.0f)	barrel_p = 0.0f;
		bool is_collise1 = CheckCollision(box_man.max_x, box_man.max_y, box_man.max_z,
			box_man.min_x, box_man.min_y, box_man.min_z,
			barrel.max_x, barrel.max_y, barrel.max_z,
			barrel.min_x, barrel.min_y, barrel.min_z,
			x_position, y_position, z_position,
			0.5f, -0.1f, -13.0f + barrel_p);
		bool is_collise2 = CheckCollision(box_man.max_x, box_man.max_y, box_man.max_z,
			box_man.min_x, box_man.min_y, box_man.min_z,
			barrel.max_x, barrel.max_y, barrel.max_z,
			barrel.min_x, barrel.min_y, barrel.min_z,
			x_position, y_position, z_position,
			0.5f, -0.1f, -15.0f + barrel_p);
		bool is_collise3 = CheckCollision(box_man.max_x, box_man.max_y, box_man.max_z,
			box_man.min_x, box_man.min_y, box_man.min_z,
			barrel.max_x, barrel.max_y, barrel.max_z,
			barrel.min_x, barrel.min_y, barrel.min_z,
			x_position, y_position, z_position,
			0.5f, -0.1f, -18.0f + barrel_p);

		bool is_collise = false;
		if (is_collise1|| is_collise2|| is_collise3) {
			score1 = -1;
			is_collise = true;
		}
		else {
			is_collise = false;
		}
	
		if (score1 == -1 && is_collise==false) {
			score1 = 0;
			score -= 1;
		}
		std::string scorestr = std::to_string(score);
		scorestr = "score:" + scorestr;
		cout << scorestr << endl;
		// render
		// ------
		glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, 1200, 900);
		lightingShader.use();
		lightingShader.setVec3("light.position", lightPos);
		lightingShader.setVec3("viewPos", camera1.Position);

		// light properties
		lightingShader.setVec3("light.ambient", 0.2f, 0.2f, 0.2f);
		lightingShader.setVec3("light.diffuse", 0.3f, 0.3f, 0.3f);
		lightingShader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

		// material properties
		lightingShader.setFloat("material.shininess", 64.0f);

		//projection view and model matrix
		glm::mat4 projectionMatrix = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix = camera1.GetViewMatrix();
		glm::mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;
		ourShader.setMat4("projection", projectionMatrix);
		ourShader.setMat4("view", viewMatrix);
		//environment model
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(1.55f, -24.05f, 19.0f));
		model = glm::scale(model, glm::vec3(0.001f,0.001f,0.001f));	
		model = glm::rotate(model, 0.0f, glm::vec3(0, 1, 0));
		ourShader.setMat4("model", model);
		ourModel.Draw(lightingShader);
		//lantern
		lightingShader.use();
		lightingShader.setVec3("light.position", lightPos);
		lightingShader.setVec3("viewPos", camera1.Position);

		// light properties
		lightingShader.setVec3("light.ambient", 0.2f, 0.2f, 0.2f);
		lightingShader.setVec3("light.diffuse", 0.3f, 0.3f, 0.3f);
		lightingShader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

		// material properties
		lightingShader.setFloat("material.shininess", 64.0f);
		
		// view/projection transformations
		glm::mat4 projection = glm::perspective(camera_zoomin, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera1.GetViewMatrix();
		lightingShader.setMat4("projection", projection);
		lightingShader.setMat4("view", view);
		// world transformation
		glm::mat4 lanternvec = glm::mat4(1.0f);
		lanternvec = glm::translate(lanternvec, glm::vec3(0.8f, -0.5f, 0.0f));
		lanternvec = glm::scale(lanternvec, glm::vec3(0.05f, 0.05f, 0.05f));
		lanternvec = glm::rotate(lanternvec, 180.0f, glm::vec3(0, 1, 0));
		ourShader.setMat4("model", lanternvec);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap);
		// bind specular map
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, specularMap);
		lantern.Draw(lightingShader);

		//barrel1
		glm::mat4 barrelvec11 = glm::mat4(1.0f);
		barrelvec11 = glm::translate(barrelvec11, glm::vec3(0.5f, -0.1f, -13.0f+barrel_p));
		barrelvec11 = glm::scale(barrelvec11, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec11 = glm::rotate(barrelvec11, barrel_r, glm::vec3(1, 0, 0));
		barrelvec11 = glm::rotate(barrelvec11, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec11);
		glm::mat4 projectionMatrix_barrel1 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix_barrel1 = camera1.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel1);
		lightingShader.setMat4("view", viewMatrix_barrel1);
		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//barrel12
		glm::mat4 barrelvec12 = glm::mat4(1.0f);
		barrelvec12 = glm::translate(barrelvec12, glm::vec3(0.5f, -0.1f, -15.0f + barrel_p));
		barrelvec12 = glm::scale(barrelvec12, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec12 = glm::rotate(barrelvec12, barrel_r, glm::vec3(1, 0, 0));
		barrelvec12 = glm::rotate(barrelvec12, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec12);

		glm::mat4 projectionMatrix_barrel12 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);
		glm::mat4 viewMatrix_barrel12 = camera1.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel12);
		lightingShader.setMat4("view", viewMatrix_barrel12);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//barrel13
		glm::mat4 barrelvec13 = glm::mat4(1.0f);
		barrelvec13 = glm::translate(barrelvec13, glm::vec3(0.5f, -0.1f, -18.0f + barrel_p));
		barrelvec13 = glm::scale(barrelvec13, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec13 = glm::rotate(barrelvec13, barrel_r, glm::vec3(1, 0, 0));
		barrelvec13 = glm::rotate(barrelvec13, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec13);
		glm::mat4 projectionMatrix_barrel13 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix_barrel13 = camera1.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel13);
		lightingShader.setMat4("view", viewMatrix_barrel13);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//text
		RenderText(textShader, scorestr, 25.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.9f, 0.7f));
		//charactor player
		glm::mat4 modelMatrix(1.0f);
		modelMatrix = glm::translate(modelMatrix, glm::vec3(x_position, y_position, z_position));
		modelMatrix = glm::scale(modelMatrix, glm::vec3(0.05f, 0.05f, 0.05f));
		modelMatrix = glm::rotate(modelMatrix, 180.0f, glm::vec3(0, 0, 1));
		modelMatrix = glm::rotate(modelMatrix, rotate_step, glm::vec3(0, 1, 0));

		getPose(animation, skeleton, elapsedTime, currentPose, identity, globalInverseTransform);

		glUseProgram(shader);
		glUniformMatrix4fv(viewProjectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
		glUniformMatrix4fv(boneMatricesLocation, boneCount, GL_FALSE, glm::value_ptr(currentPose[0]));

		glBindVertexArray(vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseTexture);
		glUniform1i(textureLocation, 0);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

		//skybox
		// draw skybox as last
		glDepthFunc(GL_LEQUAL);
		skyboxShader.use();
		glm::mat4 view_sky = glm::mat4(glm::mat3(camera1.GetViewMatrix())); // remove translation from the view matrix
		skyboxShader.setMat4("view", view_sky);
		skyboxShader.setMat4("projection", projection);
		// skybox cube
		glBindVertexArray(skyboxVAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		glBindVertexArray(0);
		glDepthFunc(GL_LESS); 
		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		/*glfwSwapBuffers(window);
		glfwPollEvents();*/

		
		// render
		// ------
		glViewport(800, 600, 400, 300);
		camera2.changePosition(rotate_step, x_position, cameraHeight, z_position);
		lightingShader.use();
		lightingShader.setVec3("light.position", lightPos);
		lightingShader.setVec3("viewPos", camera2.Position);

		// light properties
		lightingShader.setVec3("light.ambient", 0.2f, 0.2f, 0.2f);
		lightingShader.setVec3("light.diffuse", 0.3f, 0.3f, 0.3f);
		lightingShader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

		// material properties
		lightingShader.setFloat("material.shininess", 64.0f);

		//projection view and model matrix
		glm::mat4 projectionMatrix1 = glm::perspective(camera_zoomin, (float)350 / 250, 0.01f, 100.0f);

		glm::mat4 viewMatrix1 = camera2.GetViewMatrix();
		glm::mat4 viewProjectionMatrix1 = projectionMatrix1 * viewMatrix1;
		ourShader.setMat4("projection", projectionMatrix1);
		ourShader.setMat4("view", viewMatrix1);
		//environment model
		glm::mat4 model1 = glm::mat4(1.0f);
		model1 = glm::translate(model1, glm::vec3(1.55f, -24.05f, 19.0f));
		model1 = glm::scale(model1, glm::vec3(0.001f, 0.001f, 0.001f));
		model1 = glm::rotate(model1, 0.0f, glm::vec3(0, 1, 0));
		ourShader.setMat4("model", model1);
		ourModel.Draw(lightingShader);
		//lantern
		lightingShader.use();
		lightingShader.setVec3("light.position", lightPos);
		lightingShader.setVec3("viewPos", camera2.Position);

		// light properties
		lightingShader.setVec3("light.ambient", 0.2f, 0.2f, 0.2f);
		lightingShader.setVec3("light.diffuse", 0.3f, 0.3f, 0.3f);
		lightingShader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

		// material properties
		lightingShader.setFloat("material.shininess", 64.0f);

		 //view/projection transformations
		glm::mat4 projection1 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view1 = camera2.GetViewMatrix();
		lightingShader.setMat4("projection", projection1);
		lightingShader.setMat4("view", view1);
		// world transformation
		glm::mat4 lanternvec1 = glm::mat4(1.0f);
		lanternvec1 = glm::translate(lanternvec1, glm::vec3(0.8f, -0.5f, 0.0f));
		lanternvec1 = glm::scale(lanternvec1, glm::vec3(0.05f, 0.05f, 0.05f));
		lanternvec1 = glm::rotate(lanternvec1, 90.0f, glm::vec3(0, 1, 0));
		ourShader.setMat4("model", lanternvec1);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap);
		// bind specular map
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, specularMap);
		lantern.Draw(lightingShader);
		//barrel1
		glm::mat4 barrelvec21 = glm::mat4(1.0f);
		barrelvec21 = glm::translate(barrelvec21, glm::vec3(0.5f, -0.1f, -13.0f + barrel_p));
		barrelvec21 = glm::scale(barrelvec21, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec21 = glm::rotate(barrelvec21, barrel_r, glm::vec3(1, 0, 0));
		barrelvec21 = glm::rotate(barrelvec21, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec21);
		glm::mat4 projectionMatrix_barrel2 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix_barrel2 = camera2.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel2);
		lightingShader.setMat4("view", viewMatrix_barrel2);
		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//barrel12
		glm::mat4 barrelvec22 = glm::mat4(1.0f);
		barrelvec22 = glm::translate(barrelvec22, glm::vec3(0.5f, -0.1f, -15.0f + barrel_p));
		barrelvec22 = glm::scale(barrelvec22, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec22 = glm::rotate(barrelvec22, barrel_r, glm::vec3(1, 0, 0));
		barrelvec22 = glm::rotate(barrelvec22, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec22);
		glm::mat4 projectionMatrix_barrel22 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix_barrel22 = camera2.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel22);
		lightingShader.setMat4("view", viewMatrix_barrel22);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//barrel13
		glm::mat4 barrelvec23 = glm::mat4(1.0f);
		barrelvec23 = glm::translate(barrelvec23, glm::vec3(0.5f, -0.1f, -18.0f + barrel_p));
		barrelvec23 = glm::scale(barrelvec23, glm::vec3(0.01f, 0.01f, 0.01f));
		barrelvec23 = glm::rotate(barrelvec23, barrel_r, glm::vec3(1, 0, 0));
		barrelvec23 = glm::rotate(barrelvec23, 90.0f, glm::vec3(0, 0, 1));
		lightingShader.setMat4("model", barrelvec23);
		glm::mat4 projectionMatrix_barrel23 = glm::perspective(camera_zoomin, (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);

		glm::mat4 viewMatrix_barrel23 = camera2.GetViewMatrix();
		lightingShader.setMat4("projection", projectionMatrix_barrel23);
		lightingShader.setMat4("view", viewMatrix_barrel23);

		// bind diffuse map
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseMap_barrel);
		barrel.Draw(lightingShader);
		//model charactor player
		glm::mat4 modelMatrix1(1.0f);
		modelMatrix1 = glm::translate(modelMatrix1, glm::vec3(x_position, y_position, z_position));
		modelMatrix1 = glm::scale(modelMatrix1, glm::vec3(0.05f, 0.05f, 0.05f));
		modelMatrix1 = glm::rotate(modelMatrix1, 180.0f, glm::vec3(0, 0, 1));
		modelMatrix1 = glm::rotate(modelMatrix1, rotate_step, glm::vec3(0, 1, 0));

		getPose(animation, skeleton, elapsedTime, currentPose, identity, globalInverseTransform);

		glUseProgram(shader);
		glUniformMatrix4fv(viewProjectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix1));
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix1));
		glUniformMatrix4fv(boneMatricesLocation, boneCount, GL_FALSE, glm::value_ptr(currentPose[0]));

		glBindVertexArray(vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuseTexture);
		glUniform1i(textureLocation, 0);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

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
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		current_speed = -run_speed;
	}else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {	
		current_speed = run_speed;
	}
	else
	{
		current_speed = 0.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
		current_rotate_speed = -rotate_speed;
		translat_step2 -= 0.003f;
	}else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		translat_step2 += 0.003f;
		current_rotate_speed = rotate_speed;
	}
	else
	{
		current_rotate_speed = 0.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		if (!isOnSky) {
			jump_upward_speed = jump_power;
			/*cout << y_position << std::endl;*/
			isOnSky = true;
		}
	}
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera1.ProcessMouseScroll(yoffset);
}
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera1.ProcessMouseMovement(xoffset, yoffset);
}
unsigned int loadCubemap(std::string faces[])
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

	int width, height, nrChannels;
	for (unsigned int i = 0; i < 6; i++)
	{
		unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
		cout << faces[i] << std::endl;
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return textureID;
};
GLboolean CheckCollision(float x_max1, float y_max1,float z_max1,
	float x_min1, float y_min1, float z_min1,
	float x_max2, float y_max2, float z_max2,
	float x_min2, float y_min2, float z_min2,
	float x1_pos, float y1_pos, float z1_pos, 
	float x2_pos, float y2_pos, float z2_pos) // AABB - AABB collision
{
	float center1z = (float)((z_max1 - z_min1) / 2);
	float center1x = (float)((x_max1 - x_min1) / 2);
	float center1y = (float)((y_max1 - y_min1) / 2);
	float center2z = (float)((z_max2 - z_min2) / 2);
	float center2x = (float)((x_max2 - x_min2) / 2);
	float center2y = (float)((y_max2 - y_min2) / 2);
	float ditances = glm::sqrt(pow(x2_pos - x1_pos, 2) + pow(y2_pos - y1_pos, 2) + pow(z2_pos - z1_pos, 2));
	bool collision = ditances < 0.3f;

	return collision;
};
void RenderText(Shader &shader, std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color)
{
	shader.use();
	glUniform3f(glGetUniformLocation(shader.ID, "textColor"), color.x, color.y, color.z);
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(VAO);

	// Iterate through all characters
	std::string::const_iterator c;
	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = Characters[*c];

		GLfloat xpos = x + ch.Bearing.x * scale;
		GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

		GLfloat w = ch.Size.x * scale;
		GLfloat h = ch.Size.y * scale;
		// Update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos,     ypos,       0.0, 1.0 },
			{ xpos + w, ypos,       1.0, 1.0 },

			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos + w, ypos,       1.0, 1.0 },
			{ xpos + w, ypos + h,   1.0, 0.0 }
		};
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.TextureID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
};