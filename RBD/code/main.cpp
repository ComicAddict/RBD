#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint> // for uint32_t
#include <limits> // for std::numeric_limits
#include <algorithm> // for std::clamp
#include <fstream> //to read SPIRV shaders
#include <array>
#include <chrono>
#include <unordered_map>

#include "shader.h"

glm::vec3 camPos = glm::vec3(5.0f, 5.0f, 0.0f);
glm::vec3 camFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 camUp = glm::vec3(0.0f, 0.0f, 1.0f);
float sensitivity = 5.0f;
bool focused = false;

bool firstMouse = true;
float yaw = -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch = 0.0f;
float lastX = 1920.0f / 2.0;
float lastY = 1080.0 / 2.0;
float fov = 45.0f;

float deltaTimeFrame = .0f;
float lastFrame = .0f;
bool timeToSimulate = false;

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;

	bool operator==(const Vertex& other) const {
		return pos == other.pos;
	}
};

struct VertexData {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;

	bool operator==(const Vertex& other) const {
		return pos == other.pos;
	}
};

struct Edge {
	int v1;
	int v2;
};

struct Face {
	int v1;
	int v2;
	int v3;
};

struct CollResp1 {
	int obj1, obj2;
	int vertex, v1, v2, v3;
	glm::vec3 normal;
};

struct CollResp2 {
	int obj1, obj2;
	int v1, v2, v3, v4;
	glm::vec3 normal;
};

struct State {
	glm::vec3 pos;
	glm::vec3 vel;
	glm::vec3 rot;
	glm::vec3 w;
};

struct Impulse {
	glm::vec3 pos;
	glm::vec3 dir;
};

struct Object {
	State s;
	State ps;
	int index;
	std::string model_path;
	unsigned int vao, vbo, ebo;
	std::vector<Vertex> vertices;
	std::vector<VertexData> drawData;
	std::vector<uint32_t> indices;
	std::vector<Edge> edges;
	std::vector<Face> faces;
	std::vector<Impulse> impulses;
	std::vector<float> masses;
	bool dynamic;
};

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos)) >> 1);
		}
	};
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	if (focused) {

		float xpos = static_cast<float>(xposIn);
		float ypos = static_cast<float>(yposIn);

		if (firstMouse)
		{
			lastX = xpos;
			lastY = ypos;
			firstMouse = false;
		}

		float xoffset = xpos - lastX;
		float yoffset = ypos - lastY; // reversed since y-coordinates go from bottom to top
		lastX = xpos;
		lastY = ypos;

		float mouseSens = 0.2f;
		xoffset *= mouseSens;
		yoffset *= mouseSens;

		yaw -= xoffset;
		pitch -= yoffset;

		// make sure that when pitch is out of bounds, screen doesn't get flipped
		if (pitch > 89.0f)
			pitch = 89.0f;
		if (pitch < -89.0f)
			pitch = -89.0f;

		glm::vec3 front;
		front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.z = sin(glm::radians(pitch));
		camFront = glm::normalize(front);

	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int modsdouble)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {

	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		if (focused)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		else
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		focused = !focused;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{

	sensitivity += 0.2f * static_cast<float>(yoffset);
	if (sensitivity < 0) {
		sensitivity = 0.01f;
	}

	fov -= (float)yoffset;
	if (fov < 1.0f)
		fov = 1.0f;
	if (fov > 45.0f)
		fov = 45.0f;
}

void loadModel(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::string model_path) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, model_path.c_str()))
		throw std::runtime_error(warn);


	std::unordered_map<Vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes) {
		for (const auto index : shape.mesh.indices) {
			Vertex vertex{};

			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.normal = {
				attrib.normals[3 * index.normal_index + 0],
				attrib.normals[3 * index.normal_index + 1],
				attrib.normals[3 * index.normal_index + 2]
			};

			//if (uniqueVertices.count(vertex) == 0) {
			uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
			vertices.push_back(vertex);
			//}
			indices.push_back(uniqueVertices[vertex]);
		}
	}
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float camSpeed = static_cast<float>(sensitivity * deltaTimeFrame);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camPos += camSpeed * camFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camPos -= camSpeed * camFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camPos -= camSpeed * glm::normalize(glm::cross(camFront, camUp));
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camPos += camSpeed * glm::normalize(glm::cross(camFront, camUp));
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		camPos += camSpeed * camUp;
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		camPos -= camSpeed * camUp;
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

Object constructObj(std::string model_path, bool dynamic) {
	Object obj{};
	obj.dynamic = dynamic;
	obj.model_path = model_path;
	loadModel(obj.vertices, obj.indices, obj.model_path);
	for (int i = 0; i < obj.vertices.size(); i++) {
		obj.drawData.push_back({ obj.vertices[i].pos ,obj.vertices[i].normal ,obj.vertices[i].texCoord });
	}
	glGenVertexArrays(1, &obj.vao);
	glGenBuffers(1, &obj.vbo);
	glGenBuffers(1, &obj.ebo);

	glBindVertexArray(obj.vao);
	glBindBuffer(GL_ARRAY_BUFFER, obj.vbo);
	glBufferData(GL_ARRAY_BUFFER, obj.drawData.size() * sizeof(Vertex), &obj.drawData[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.indices.size() * sizeof(float), &obj.indices[0], GL_DYNAMIC_DRAW);
	obj.s.pos = glm::vec3(0.0f, 0.0f, 0.0f);
	obj.s.vel = glm::vec3(0.0f, 0.0f, 0.0f);
	obj.s.rot = glm::vec3(0.0f, 0.0f, 0.0f);
	obj.s.w = glm::vec3(0.0f, 0.0f, 0.0f);
	//if (dynamic) {
		obj.s.pos = glm::vec3(0.0f, 0.0f, 0.0f);
		obj.s.vel = glm::vec3(0.0f, 0.0f, 0.0f);
		obj.s.rot = glm::vec3(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < obj.vertices.size(); i++) {
			obj.masses.push_back(1.0f);
			obj.s.pos += obj.masses.back() * obj.vertices[i].pos;
		}
		obj.s.pos /= obj.vertices.size();
		//printf("Dynamic Object at with mass center (%f, %f, %f)\n", obj.s.pos.x, obj.s.pos.y, obj.s.pos.z);
	//}

	for (int i = 0; i < obj.indices.size() / 3; i++) {
		//printf("%i, %i\n", obj.indices[i], i);
		Face f;
		Edge e1;
		Edge e2;
		Edge e3;
		f.v1 = obj.indices[3 * i];
		f.v2 = obj.indices[3 * i + 1];
		f.v3 = obj.indices[3 * i + 2];
		e1.v1 = f.v1;
		e1.v2 = f.v2;
		e2.v1 = f.v2;
		e2.v2 = f.v3;
		e3.v1 = f.v3;
		e3.v2 = f.v1;
		obj.faces.push_back(f);
		obj.edges.push_back(e1);
		obj.edges.push_back(e2);
		obj.edges.push_back(e3);
	}
	return obj;
}

void loadObjBufferData(Object& obj) {
	glBindBuffer(GL_ARRAY_BUFFER, obj.vbo);
	glBufferData(GL_ARRAY_BUFFER, obj.drawData.size() * sizeof(Vertex), &obj.drawData[0], GL_DYNAMIC_DRAW);
}

void findDerivativeState(State& der, State& s, std::vector<Object>& objects) {
	for (size_t i = 0; i < objects.size(); i++) {
		if (objects[i].dynamic) {
			// calculate state derivative
			der.pos = s.vel;
			glm::vec3 acc = glm::vec3(0.0f, 0.0f, -1.0f);
			der.vel = acc;
			der.rot = s.w;
		}
	}
}

glm::vec3 rotateXYZ(glm::vec3& vec, glm::vec3 &rot) {
	return glm::rotateZ(glm::rotateY(glm::rotateX(vec, rot.x), rot.y), rot.z);
}

bool vertexFaceIntersection(Vertex &v, Face &f) {
	return false;
}

int main() {


	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // opengl version 3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //opengil version 3.3
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //using core profile of opengl
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	GLFWwindow* window = glfwCreateWindow(1920, 1080, "Spring Meshes", NULL, NULL);
	//glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, 1920, 1080, GLFW_DONT_CARE);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glViewport(0, 0, 1920, 1080);

	//load model
	std::vector<Object> objects;
	bool dynamic = true;
	objects.push_back(constructObj("C:\\Src\\meshes\\cube.obj", dynamic));
	objects.push_back(constructObj("C:\\Src\\meshes\\cube.obj", dynamic));
	for (int i = 0; i < 10; i++) {
		objects.push_back(constructObj("C:\\Src\\meshes\\icos1.obj", dynamic));
		objects.back().s.pos = glm::linearRand(glm::vec3(-10.f,-10.f,1.0f), glm::vec3(10.f, 10.f, 5.0f));
		objects.back().s.w = glm::linearRand(glm::vec3(-20.f, -20.f, -20.0f), glm::vec3(20.f, 20.f, 20.0f));
	}
	objects.push_back(constructObj("C:\\Src\\meshes\\plane.obj", false));
	//objects.push_back(constructObj("C:\\Src\\meshes\\suzanne.obj", true));
	for (int i = 0; i < objects.size(); i++) {
		objects[i].index = i;
	}
	printf("Constructed objects\n");
	objects[0].s.pos = glm::vec3(1.0f, 5.0f, 0.0f);
	objects[0].s.w = glm::vec3(10.0f, 5.0f, 0.0f);
	objects[0].s.rot = glm::vec3(45.0f, 0.0f, -15.0f);
	objects[1].s.pos = glm::vec3(1.0f, 5.0f, 5.0f);
	objects[1].s.rot = glm::vec3(15.0f, 45.0f, 0.0f);
	
	for (int i = 0; i < objects.size(); i++) {
		objects[i].ps = objects[i].s;
		printf("Object %i: %s\n", i, objects[i].model_path.c_str());
	}
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	Shader shader("C:\\Src\\shaders\\vertRBD.glsl", "C:\\Src\\shaders\\fragRBD.glsl");
	shader.use();
	int width, height;

	glEnable(GL_DEPTH_TEST);


	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	ImGuiViewport* viewport = ImGui::GetMainViewport();
	static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
	float lightPos[3] = {0.0f,0.0f,0.0f};
	float h = 0.01f;
	while (!glfwWindowShouldClose(window))
	{
		// time handling for input, should not interfere with this
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTimeFrame = currentFrame - lastFrame;
		lastFrame = currentFrame;
		processInput(window);

		glClearColor(0.7f, .01f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::DockSpaceOverViewport(viewport, dockspace_flags);

		glm::mat4 view = glm::lookAt(camPos, camPos + camFront, camUp);
		shader.setMat4("view", view);

		glfwGetWindowSize(window, &width, &height);

		glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 400.0f);
		glm::mat4 model = glm::mat4(1.0f);

		shader.setMat4("projection", projection);
		shader.setMat4("model", model);
		shader.setVec3("lightPos", lightPos[0], lightPos[1], lightPos[2]);
		glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
		for (size_t i = 0; i < objects.size(); i++) {
			glBindVertexArray(objects[i].vao);
			loadObjBufferData(objects[i]);
			glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(objects[i].indices.size()), GL_UNSIGNED_INT, 0);
		}
		if (timeToSimulate) {
			// integrate 
			State change{};
			for (size_t i = 0; i < objects.size(); i++) {
				if (objects[i].dynamic) {
					// calculate forces
					State tempState{};
					State k1{};
					findDerivativeState(k1, objects[i].s, objects);
					tempState.pos = objects[i].s.pos + 0.5f * deltaTimeFrame * k1.pos;
					tempState.vel = objects[i].s.vel + 0.5f * deltaTimeFrame * k1.vel;
					/*
					State k2{};
					findDerivativeState(k2, tempState, objects);
					tempState.pos = objects[i].s.pos + 0.5f * deltaTimeFrame * k2.pos;
					tempState.vel = objects[i].s.vel + 0.5f * deltaTimeFrame * k2.vel;

					State k3{};
					findDerivativeState(k3, tempState, objects);
					tempState.pos = objects[i].s.pos + deltaTimeFrame * k3.pos;
					tempState.vel = objects[i].s.vel + deltaTimeFrame * k3.vel;

					State k4{};
					findDerivativeState(k4, tempState, objects);
					tempState.pos = objects[i].s.pos + deltaTimeFrame * k4.pos;
					tempState.vel = objects[i].s.vel + deltaTimeFrame * k4.vel;
					*/
					//change.pos = deltaTimeFrame * (k1.pos + 2.0f * k2.pos + 2.0f * k3.pos + k4.pos) / 6.0f;
					//change.vel = deltaTimeFrame * (k1.vel + 2.0f * k2.vel + 2.0f * k3.vel + k4.vel) / 6.0f;
					//printf("Change of object %i p:(%f, %f, %f), v:(%f, %f, %f)\n", i, change.pos.x, change.pos.y, change.pos.z, change.vel.x, change.vel.y, change.vel.z);
					change.pos = h * k1.pos;
					change.vel = h * k1.vel;
					change.rot = h * k1.rot;
					change.w = h * k1.w;
					objects[i].s.pos += change.pos;
					objects[i].s.vel += change.vel;
					objects[i].s.rot += change.rot;
					objects[i].s.w += change.w;
					//printf("Object %i p:(%f, %f, %f), v:(%f, %f, %f)\n", i, objects[i].s.pos.x, objects[i].s.pos.y, objects[i].s.pos.z, objects[i].s.vel.x, objects[i].s.vel.y, objects[i].s.vel.z);
					for (size_t j = 0; j < objects[i].vertices.size(); j++) {
						//objects[i].drawData[j].pos = objects[i].s.pos + objects[i].vertices[j].pos;
						//printf("Vertices of vertex %i: (%f, %f, %f)\n", j, objects[i].vertices[j].pos.x, objects[i].vertices[j].pos.y, objects[i].vertices[j].pos.z);
					}
				}
			}
			// find the collisions
			for (size_t i = 0; i < objects.size(); i++) {
				for (size_t j = i + 1; j < objects.size(); j++) {

					// vertex face
					for (Face f : objects[j].faces) {
						//printf("before norm\n");
						glm::vec3 norm = rotateXYZ(objects[j].vertices[f.v1].normal, objects[j].s.rot);
						//printf("after norm\n");
						glm::vec3 v1 = objects[j].s.pos + rotateXYZ(objects[j].vertices[f.v1].pos, objects[j].s.rot);
						glm::vec3 v2 = objects[j].s.pos + rotateXYZ(objects[j].vertices[f.v2].pos, objects[j].s.rot);
						glm::vec3 v3 = objects[j].s.pos + rotateXYZ(objects[j].vertices[f.v3].pos, objects[j].s.rot);
						//printf("after faces\n");

						glm::vec3 v1_prev = objects[j].drawData[f.v1].pos;
						glm::vec3 v2_prev = objects[j].drawData[f.v2].pos;
						glm::vec3 v3_prev = objects[j].drawData[f.v3].pos;
						for (size_t k = 0; k < objects[i].vertices.size(); k++) {
							glm::vec3 v = objects[i].s.pos + rotateXYZ(objects[i].vertices[k].pos, objects[i].s.rot);
							glm::vec3 v_prev = objects[i].drawData[k].pos;
							float side1 = glm::dot(v - v1, norm);
							float side2 = glm::dot(v_prev - v1_prev, norm);
							if (signbit(side1) != signbit(side2)) {
								// change in sides, check point inclusion by projection
								if (norm.x > norm.y && norm.x > norm.z) {
									// yz project
									float s1 = (v2.y - v1.y) * (v.z - v1.z) - (v2.z - v1.z) * (v.y - v1.y);
									float s2 = (v3.y - v2.y) * (v.z - v2.z) - (v3.z - v2.z) * (v.y - v2.y);
									float s3 = (v1.y - v3.y) * (v.z - v3.z) - (v1.z - v3.z) * (v.y - v3.y);
									if (signbit(s1) == signbit(s2) && signbit(s2) == signbit(s3)) {
										//inside the polygon, mark intersection
										glm::vec3 p = v + glm::dot(v - v1, norm) * norm;
										objects[i].impulses.push_back({ p, norm });
										objects[j].impulses.push_back({ p, norm });
									}
								}
								else if (norm.y > norm.z && norm.y > norm.x) {
									// xz project
									float s1 = (v2.x - v1.x) * (v.z - v1.z) - (v2.z - v1.z) * (v.x - v1.x);
									float s2 = (v3.x - v2.x) * (v.z - v2.z) - (v3.z - v2.z) * (v.x - v2.x);
									float s3 = (v1.x - v3.x) * (v.z - v3.z) - (v1.z - v3.z) * (v.x - v3.x);
									if (signbit(s1) == signbit(s2) && signbit(s2) == signbit(s3)) {
										glm::vec3 p = v+glm::dot(v - v1, norm)*norm;
										objects[i].impulses.push_back({p, norm});
										objects[j].impulses.push_back({ p, norm });
									}
								}
								else {
									// xy project
									float s1 = (v2.x - v1.x) * (v.y - v1.y) - (v2.y - v1.y) * (v.x - v1.x);
									float s2 = (v3.x - v2.x) * (v.y - v2.y) - (v3.y - v2.y) * (v.x - v2.x);
									float s3 = (v1.x - v3.x) * (v.y - v3.y) - (v1.y - v3.y) * (v.x - v3.x);
									if (signbit(s1) == signbit(s2) && signbit(s2) == signbit(s3)) {
										glm::vec3 p = v + glm::dot(v - v1, norm) * norm;
										objects[i].impulses.push_back({ p, norm });
										objects[j].impulses.push_back({ p, norm });
									}
								}
							}
						}
					}
					/*
					//edge edge is not working, might get back on it later on
					for (Edge e1 : objects[i].edges) {
						for (Edge e2 : objects[j].edges) {
							glm::vec3 v1 = objects[i].vertices[e1.v2].pos - objects[i].vertices[e1.v1].pos;
							//printf("orientation of e1 before rotation (%f, %f, %f)", v1.x, v1.y, v1.z);
							v1 = rotateXYZ(v1, objects[i].s.rot);
							//printf("orientation of e1 after rotation (%f, %f, %f)", v1.x, v1.y, v1.z);
							glm::vec3 v2 = objects[j].vertices[e2.v2].pos - objects[j].vertices[e2.v1].pos;
							v2 = rotateXYZ(v2, objects[j].s.rot);

							glm::vec3 v1_prev = objects[i].drawData[e1.v1].pos - objects[i].drawData[e1.v2].pos;
							v1_prev = rotateXYZ(v1_prev, objects[i].ps.rot);
							glm::vec3 v2_prev = objects[j].drawData[e2.v1].pos - objects[j].drawData[e2.v2].pos;
							v2_prev = rotateXYZ(v2_prev, objects[j].ps.rot);

							glm::vec3 n = glm::cross(v1, v2);
							if(glm::length(n) < 0.01f)
								glm::vec3 n = glm::cross(v1, v1+glm::vec3(0.1f,0.9f,0.0f));
							glm::vec3 dist1 = objects[j].s.pos + rotateXYZ(objects[j].vertices[e2.v1].pos, objects[j].s.rot) - objects[i].s.pos - rotateXYZ(objects[i].vertices[e1.v1].pos, objects[i].s.rot);
							glm::vec3 dist2 = objects[j].drawData[e2.v2].pos - objects[i].drawData[e1.v1].pos;

							if (signbit(glm::dot(n, dist1)) != signbit(glm::dot(n, dist2))) {
								printf("Edge Edge happened between object %i and %i\n", i , j);
								printf("Results of dots %f and %f\n", glm::dot(n, dist1), glm::dot(n, dist2));
								//objects[i].dynamic = false;
								//objects[j].dynamic = false;
							}

						}
					}*/
				}
			}
			
			//handle the collision
			for (size_t i = 0; i < objects.size(); i++) {
				objects[i].ps = objects[i].s;
			}
			
		}

		//update the positions
		for (size_t i = 0; i < objects.size(); i++) {
			for (size_t j = 0; j < objects[i].vertices.size(); j++) {
				objects[i].drawData[j].pos = objects[i].s.pos + glm::rotateZ(glm::rotateY(glm::rotateX(objects[i].vertices[j].pos, objects[i].s.rot.x), objects[i].s.rot.y), objects[i].s.rot.z);
				objects[i].drawData[j].normal = rotateXYZ(objects[i].vertices[j].normal, objects[i].s.rot);
				//printf("Vertices of vertex %i: (%f, %f, %f)\n", j, objects[i].vertices[j].pos.x, objects[i].vertices[j].pos.y, objects[i].vertices[j].pos.z);
			}
		}
		
		ImGui::Begin("Simulation Settings");
		if (ImGui::Button("Start Simulation")) {
			timeToSimulate = true;
		}

		if (ImGui::Button("Stop Simulation")) {
			timeToSimulate = false;
		}
		ImGui::End();

		ImGui::Begin("Integrator Settings");
		ImGui::DragFloat("Time Step", &h, 0.001);
		ImGui::End();

		ImGui::Begin("Render Settings");
		ImGui::DragFloat3("Light Pos", lightPos, 0.1f);
		ImGui::End();

		ImGui::Begin("Outliner");
		ImGui::End();

		ImGui::Render();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return 0;
}