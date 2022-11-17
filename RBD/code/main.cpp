#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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

struct Spring {
	int v1, v2;
	float baselength;
	float k;
	float d;
	glm::vec3 direction;
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
	std::vector<glm::vec3> pos;
	std::vector<glm::vec3> vel;
};

struct Object {
	State s;
	int index;
	std::string model_path;
	unsigned int vao, vbo, ebo;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Edge> edges;
	std::vector<Face> faces;
	std::vector<glm::vec3> forces;
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

			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vertex);
			}
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
	glGenVertexArrays(1, &obj.vao);
	glGenBuffers(1, &obj.vbo);
	glGenBuffers(1, &obj.ebo);

	glBindVertexArray(obj.vao);
	glBindBuffer(GL_ARRAY_BUFFER, obj.vbo);
	glBufferData(GL_ARRAY_BUFFER, obj.vertices.size() * sizeof(Vertex), &obj.vertices[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.indices.size() * sizeof(float), &obj.indices[0], GL_DYNAMIC_DRAW);
	if (dynamic) {
		for (int i = 0; i < obj.vertices.size(); i++) {
			for (int j = i + 1; j < obj.vertices.size(); j++) {
				glm::vec3 dist = obj.vertices[i].pos - obj.vertices[j].pos;
				obj.springs.push_back({ i, j, glm::length(dist), 8.0f,0.1f, glm::normalize(dist) });
			}
			obj.masses.push_back(1.0f);
			obj.forces.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
			obj.s.pos.push_back(obj.vertices[i].pos);
			obj.s.vel.push_back(glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f)));
		}
	}

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
	glBufferData(GL_ARRAY_BUFFER, obj.vertices.size() * sizeof(Vertex), &obj.vertices[0], GL_DYNAMIC_DRAW);
}

void findDerivativeState(State& der, State& s, std::vector<Object>& objects) {
	for (size_t i = 0; i < objects.size(); i++) {
		if (objects[i].dynamic) {
			// calculate forces
			for (size_t j = 0; j < objects[i].springs.size(); j++) {
				glm::vec3 dist = s.pos[objects[i].springs[j].v1] - s.pos[objects[i].springs[j].v2];
				glm::vec3 distnorm = glm::normalize(dist);
				//printf("displacement amount of spring between %i and %i is %f\n", objects[i].springs[j].v1, objects[i].springs[j].v2, (glm::length(dist) - objects[i].springs[j].baselength));
				glm::vec3 sf = objects[i].springs[j].k * (glm::length(dist) - objects[i].springs[j].baselength) * distnorm;
				glm::vec3 df = objects[i].springs[j].d * (glm::dot(s.vel[objects[i].springs[j].v1] - s.vel[objects[i].springs[j].v2], distnorm)) * distnorm;
				objects[i].forces[objects[i].springs[j].v1] -= sf + df;
				objects[i].forces[objects[i].springs[j].v2] += sf + df;
			}
			//adding gravity
			for (size_t j = 0; j < objects[i].vertices.size(); j++) {
				objects[i].forces[j] += glm::vec3(0.0, 0.0, -1.0f) * objects[i].masses[j];
			}
			// calculate state derivative
			der.pos = s.vel;
			for (size_t j = 0; j < objects[i].vertices.size(); j++) {
				glm::vec3 acc = objects[i].forces[j] / objects[i].masses[j];
				//printf("acc of point %i is (%f, %f, %f)\n", j, acc.x, acc.y, acc.z);
				der.vel.push_back(acc);
				objects[i].forces[j] = glm::vec3(0.0f, 0.0f, 0.0f);
			}
		}
	}
}

float hpSign(glm::vec2 p1, glm::vec2 p2, glm::vec2 p3) {
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

void vertexFaceIntersection(std::vector<CollResp1>& resp, Object& obj1, Object& obj2, State change) {
	for (int i = 0; i < obj1.vertices.size(); i++) {
		Vertex v = obj1.vertices[i];
		for (int j = 0; j < obj2.faces.size(); j++) {
			Face f = obj2.faces[j];
			glm::vec3 p_prev = obj1.s.pos[i];
			glm::vec3 norm = glm::normalize(glm::cross(obj2.vertices[f.v1].pos - obj2.vertices[f.v2].pos, obj2.vertices[f.v2].pos - obj2.vertices[f.v3].pos));
			float d = glm::dot(v.pos - obj2.vertices[f.v1].pos, norm);
			obj1.s.pos[i] += change.pos[i] * deltaTimeFrame;
			obj1.s.vel[i] += change.vel[i] * deltaTimeFrame;
			float dn = glm::dot(v.pos - obj2.vertices[f.v1].pos, norm);
			if (signbit(d) != signbit(dn)) {
				printf("coll level 1\n");
				//collision may happened need to check projections
				//for xy
				glm::vec3 collP = p_prev + change.pos[i] * deltaTimeFrame * (abs(d) / (abs(d) + abs(dn)));
				obj1.s.pos[i] = collP + norm * 0.01f;
				glm::vec3 vn = glm::dot(obj1.s.vel[i], norm) * norm;

				glm::vec3 vt = obj1.s.vel[i] - vn;
				obj1.s.vel[i] = -vn + vt;
				obj1.vertices[i].pos = obj1.s.pos[i];
				//float e1 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(obj2.vertices[f.v1].pos.y, obj2.vertices[f.v1].pos.z), glm::vec2(obj2.vertices[f.v2].pos.y, obj2.vertices[f.v2].pos.z));
				//float e2 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(obj2.vertices[f.v2].pos.y, obj2.vertices[f.v2].pos.z), glm::vec2(obj2.vertices[f.v3].pos.y, obj2.vertices[f.v3].pos.z));
				//float e3 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(obj2.vertices[f.v3].pos.y, obj2.vertices[f.v3].pos.z), glm::vec2(obj2.vertices[f.v1].pos.y, obj2.vertices[f.v1].pos.z));
				//resp.push_back({ obj1.index, obj2.index, i, f.v1, f.v2, f.v3, norm });
				//if ((signbit(e1) == signbit(e2) && signbit(e2) == signbit(e3))) {
					//printf("coll level 2\n");
					//printf("coll happened\n");


				//}
			}
		}
	}
	/*
	glm::vec3 p_prev = pgen1.pgpus[i].p;
	float d = glm::dot(pgen1.pgpus[i].p - tri[0], norm);
	pgen1.pgpus[i].p += pgen1.pcpus[i].v * deltaTimeFrame;
	float dn = glm::dot(pgen1.pgpus[i].p - tri[0], norm);
	pgen1.pcpus[i].v += g * glm::vec3(0.0, 0.0, -1.0) * deltaTimeFrame;
	pgen1.pcpus[i].v = (1 - lorenzFac * 0.01f) * pgen1.pcpus[i].v + lorenzFac * 0.01f * velocity;
	if (signbit(d) != signbit(dn)) {
		printf("coll level 1\n");
		//collision may happened need to check projections
		glm::vec3 collP = p_prev + pgen1.pcpus[i].v * deltaTimeFrame * (abs(d) / (abs(d) + abs(dn)));
		//for xy
		float e1 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(tri[0].y, tri[0].z), glm::vec2(tri[1].y, tri[1].z));
		float e2 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(tri[1].y, tri[1].z), glm::vec2(tri[2].y, tri[2].z));
		float e3 = hpSign(glm::vec2(collP.y, collP.z), glm::vec2(tri[2].y, tri[2].z), glm::vec2(tri[0].y, tri[0].z));

		if (signbit(e1) == signbit(e2) && signbit(e2) == signbit(e3)) {
			printf("coll level 2\n");
			printf("coll happened\n");
			pgen1.pgpus[i].p -= 2.0f * glm::dot(pgen1.pgpus[i].p, norm) * norm;
			glm::vec3 vn = glm::dot(pgen1.pcpus[i].v, norm) * norm;

			glm::vec3 vt = pgen1.pcpus[i].v - vn;
			pgen1.pcpus[i].v = -vn + vt;
		}
	}
	*/
}

void edgeEdgeIntersection(std::vector<CollResp2> resp, Object obj1, Object obj2) {
	for (Edge e1 : obj1.edges) {
		for (Edge e2 : obj2.edges) {

		}
	}
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
	objects.push_back(constructObj("C:\\Users\\Tolga YILDIZ\\springmeshes\\springmeshes\\mesh\\suzanne.obj", true));
	objects.push_back(constructObj("C:\\Users\\Tolga YILDIZ\\springmeshes\\springmeshes\\mesh\\plane.obj", false));
	for (int i = 0; i < objects.size(); i++) {
		objects[i].index = i;
	}
	//objects.push_back(constructObj("C:\\Users\\Tolga YILDIZ\\springmeshes\\springmeshes\\mesh\\sphere.obj"));


	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	Shader shader("C:\\Users\\Tolga YILDIZ\\springmeshes\\springmeshes\\shaders\\vert.glsl", "C:\\Users\\Tolga YILDIZ\\springmeshes\\springmeshes\\shaders\\frag.glsl");
	shader.use();
	int width, height;

	glEnable(GL_DEPTH_TEST);


	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");



	while (!glfwWindowShouldClose(window))
	{
		// time handling for input, should not interfere with this
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTimeFrame = currentFrame - lastFrame;
		lastFrame = currentFrame;
		processInput(window);

		glClearColor(0.1f, .7f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		glm::mat4 view = glm::lookAt(camPos, camPos + camFront, camUp);
		shader.setMat4("view", view);

		glfwGetWindowSize(window, &width, &height);

		glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 400.0f);
		glm::mat4 model = glm::mat4(1.0f);

		shader.setMat4("projection", projection);
		shader.setMat4("model", model);
		for (size_t i = 0; i < objects.size(); i++) {
			glBindVertexArray(objects[i].vao);
			loadObjBufferData(objects[i]);
			glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(objects[i].indices.size()), GL_UNSIGNED_INT, 0);
		}
		State change{};
		change.pos.reserve(objects[0].s.pos.size());
		change.vel.reserve(objects[0].s.pos.size());
		for (size_t i = 0; i < objects.size(); i++) {
			if (objects[i].dynamic) {
				// calculate forces
				State tempState{};
				State k1{};
				findDerivativeState(k1, objects[i].s, objects);
				for (size_t j = 0; j < k1.pos.size(); j++) {
					tempState.pos.push_back(objects[i].s.pos[j] + 0.5f * deltaTimeFrame * k1.pos[j]);
					tempState.vel.push_back(objects[i].s.vel[j] + 0.5f * deltaTimeFrame * k1.vel[j]);
				}
				//printf("Sizes of k1 %i, %i", k1.pos.size(), k1.vel.size());
				State k2{};
				findDerivativeState(k2, tempState, objects);

				tempState.pos.clear();
				tempState.vel.clear();
				for (size_t j = 0; j < k1.pos.size(); j++) {
					tempState.pos.push_back(objects[i].s.pos[j] + 0.5f * deltaTimeFrame * k2.pos[j]);
					tempState.vel.push_back(objects[i].s.vel[j] + 0.5f * deltaTimeFrame * k2.vel[j]);
				}
				State k3{};
				findDerivativeState(k3, tempState, objects);
				tempState.pos.clear();
				tempState.vel.clear();
				for (size_t j = 0; j < k1.pos.size(); j++) {
					tempState.pos.push_back(objects[i].s.pos[j] + deltaTimeFrame * k3.pos[j]);
					tempState.vel.push_back(objects[i].s.vel[j] + deltaTimeFrame * k3.vel[j]);
				}
				State k4{};
				findDerivativeState(k4, tempState, objects);
				for (size_t j = 0; j < k1.pos.size(); j++) {
					change.pos.push_back(deltaTimeFrame * (k1.pos[j] + 2.0f * k2.pos[j] + 2.0f * k3.pos[j] + k4.pos[j]) / 6.0f);
					change.vel.push_back(deltaTimeFrame * (k1.vel[j] + 2.0f * k2.vel[j] + 2.0f * k3.vel[j] + k4.vel[j]) / 6.0f);
					//objects[i].s.pos[j] += deltaTimeFrame * (k1.pos[j] + 2.0f * k2.pos[j] + 2.0f * k3.pos[j] + k4.pos[j]) / 6.0f;
					//objects[i].s.vel[j] += deltaTimeFrame * (k1.vel[j] + 2.0f * k2.vel[j] + 2.0f * k3.vel[j] + k4.vel[j]) / 6.0f;
					objects[i].vertices[j].pos = objects[i].s.pos[j];
				}
			}
		}
		std::vector<CollResp1> faceVertex;
		std::vector<CollResp2> edgeEdge;
		//vertexFaceIntersection(faceVertex, objects[0], objects[1], change);
		//for (int i = 0; i < objects.size(); i++) {
			//for (int j = 0; j < objects.size(); j++) {
				//if (i != j) {

				//}
				//edgeEdgeIntersection(edgeEdge, objects[i], objects[j]);
			//}
		//}

		for (int i = 0; i < objects[0].vertices.size(); i++) {
			Vertex v = objects[0].vertices[i];
			for (int j = 0; j < objects[1].faces.size(); j++) {
				Face f = objects[1].faces[j];
				glm::vec3 p_prev = objects[0].s.pos[i];
				glm::vec3 norm = glm::normalize(glm::cross(objects[1].vertices[f.v1].pos - objects[1].vertices[f.v2].pos, objects[1].vertices[f.v2].pos - objects[1].vertices[f.v3].pos));
				float d = glm::dot(objects[0].s.pos[i] - objects[1].vertices[f.v1].pos, norm);
				objects[0].s.pos[i] += change.pos[i];
				objects[0].s.vel[i] += change.vel[i];
				float dn = glm::dot(objects[0].s.pos[i] - objects[1].vertices[f.v1].pos, norm);
				if (signbit(d) != signbit(dn)) {
					//printf("coll level 1\n");
					//collision may happened need to check projections
					//for xy
					glm::vec3 collP = p_prev + change.pos[i] * (abs(d) / (abs(d) + abs(dn)));
					objects[0].s.pos[i] += norm * 0.1f;
					glm::vec3 vn = glm::dot(objects[0].s.vel[i], norm) * norm;

					glm::vec3 vt = objects[0].s.vel[i] - vn;
					objects[0].s.vel[i] = -vn + vt;
					objects[0].vertices[i].pos = objects[0].s.pos[i];
				}
			}
		}
		//printf("out of finding intersections\n");
		//for (CollResp1 const& resp : faceVertex) {
			//printf("collided objects 1:%i,%i ; 2:%i,%i", resp.obj1, objects[resp.obj1].vertices.size(), resp.obj2, objects[resp.obj2].vertices.size());
			//if (objects[resp.obj1].dynamic && !objects[resp.obj2].dynamic) {
				//objects[resp.obj1].vertices[resp.vertex].pos += resp.normal * 0.1f;
				//objects[resp.obj1].s.pos[resp.vertex] += resp.normal * 0.1f;
				//pgen1.pgpus[i].p -= 2.0f * glm::dot(pgen1.pgpus[i].p, norm) * norm;
				//glm::vec3 vn = glm::dot(objects[resp.obj1].s.vel[resp.vertex], resp.normal) * resp.normal;

				//glm::vec3 vt = objects[resp.obj1].s.vel[resp.vertex] - vn;
				//objects[resp.obj1].s.vel[resp.vertex] = -vn;
				//printf("velocity after coll: %f, %f, %f\n", objects[resp.obj1].s.vel[resp.vertex].x, objects[resp.obj1].s.vel[resp.vertex].y, objects[resp.obj1].s.vel[resp.vertex].z);
			//}
			/*
			else if (!objects[resp.obj1].dynamic && objects[resp.obj2].dynamic) {
				objects[resp.obj2].vertices[resp.v1].pos += resp.normal / 3.0f;
				objects[resp.obj2].vertices[resp.v2].pos += resp.normal / 3.0f;
				objects[resp.obj2].vertices[resp.v3].pos += resp.normal / 3.0f;
				objects[resp.obj2].s.pos[resp.v1] += resp.normal / 3.0f;
				objects[resp.obj2].s.pos[resp.v2] += resp.normal / 3.0f;
				objects[resp.obj2].s.pos[resp.v3] += resp.normal / 3.0f;
				//pgen1.pgpus[i].p -= 2.0f * glm::dot(pgen1.pgpus[i].p, norm) * norm;
				glm::vec3 avVel = (objects[resp.obj2].s.vel[resp.v1] + objects[resp.obj2].s.vel[resp.v2] + objects[resp.obj2].s.vel[resp.v3]) / 3.0f;
				glm::vec3 vn = glm::dot(avVel, resp.normal) * resp.normal;
				glm::vec3 vt = avVel - vn;
				objects[resp.obj2].s.vel[resp.v1] = (-vn + vt) / 3.0f;
				objects[resp.obj2].s.vel[resp.v2] = (-vn + vt) / 3.0f;
				objects[resp.obj2].s.vel[resp.v3] = (-vn + vt) / 3.0f;
				//printf("this is static vs dynamic check\n");
			}
			else if (objects[resp.obj1].dynamic && objects[resp.obj2].dynamic) {

			}*/
			//}
		ImGui::Begin("Integrator Settings");
		ImGui::End();

		ImGui::Begin("Spring Settings");
		ImGui::End();


		ImGui::Render();
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