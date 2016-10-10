#include "main.h"
#include "preview.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;
float iterationTimeMs;
double totalElapsedTimeMs = 0.0;
double numPathsFinished;

int width;
int height;

static const char *phoneVsSrc =
"#version 450 core\n"
"layout(location = 0) uniform mat4 MVP;"
"layout(location = 0) in vec3 position;"
"layout(location = 1) in vec3 normal;"
"out VS_OUT"
"{"
"	vec3 pos;"
"	vec3 normal;"
"} vs_out;"
"void main()"
"{"
"	gl_Position = MVP * vec4(position, 1.f);"
"	vs_out.pos = position;"
"	vs_out.normal = normal;"
"}";

static const char *phonePsSrc =
"#version 450 core\n"
"layout(location = 1) uniform vec3 lightDir;\n"
"layout(location = 2) uniform vec3 eyePos;\n"
"in VS_OUT\n"
"{\n"
"	vec3 pos;\n"
"	vec3 normal;\n"
"} ps_in;\n"
"out vec4 final_color;\n"
"void main()\n"
"{\n"
"   const vec3 ambient = vec3(1, 1, 1);\n"
"	const vec3 specColor = vec3(1, 1, 1);\n"
"	const vec3 diffColor = vec3(1, 1, 1);\n"
"   const float Ka = 0.1;\n"
"	const float Kd = 0.5;\n"
"	const float Ks = 0.5;\n"
"	const float exponent = 20.0;\n"
"	vec3 viewDir = normalize(eyePos - ps_in.pos);\n"
"	vec3 h = normalize(lightDir + viewDir);\n"
"	float costheta = clamp(dot(normalize(ps_in.normal), h), 0.0, 1.0);\n"
"	final_color = vec4(Ka * ambient + Kd * diffColor + Ks * specColor * pow(costheta, exponent), 1.0);\n"
"}";

static const char *lineVsSrc =
"#version 450 core\n"
"layout(location = 0) uniform mat4 MVP;"
"layout(location = 0) in vec3 position;"
"void main()"
"{"
"	gl_Position = MVP * vec4(position, 1.0);"
"}";

static const char *linePsSrc =
"#version 450 core\n"
"out vec4 final_color;"
"void main()"
"{"
"	final_color = vec4(1, 0, 1, 1);"
"}";

class BVHDebug
{
public:
	BVHDebug(Scene *scene) : scene(scene) {}
	
	void run()
	{
		initWindow();
		initShaders();
		initVertexBuffers();
		glClearColor(0.f, 0.f, 0.f, 0.f);
		glEnable(GL_DEPTH_TEST);

		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();

			if (camchanged)
			{
				Camera &cam = renderState->camera;
				cameraPosition.x = zoom * sin(phi) * sin(theta);
				cameraPosition.y = zoom * cos(theta);
				cameraPosition.z = zoom * cos(phi) * sin(theta);

				cam.view = -glm::normalize(cameraPosition);
				glm::vec3 v = cam.view;
				glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
				glm::vec3 r = glm::cross(v, u);
				cam.up = glm::cross(r, v);
				cam.right = r;

				cameraPosition += cam.lookAt;
				cam.position = cameraPosition;
				camchanged = false;
			}

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			const Camera &cam = scene->state.camera;
			glm::mat4 viewMatrix = glm::lookAt(cam.position, cam.lookAt, cam.up);
			glm::mat4 projMatrix = glm::perspective(cam.fov.y,
				cam.resolution.x / float(cam.resolution.y),
				.1f, 100.f);
			glm::mat4 MVP = projMatrix * viewMatrix;
			glm::vec3 eyePos = cam.position;
			glm::vec3 lightDir = glm::normalize(glm::vec3(1.f, 1.f, 1.f));

			glUseProgram(phoneProgram);
			glBindVertexArray(modelVAO);
			glUniformMatrix4fv(0, 1, GL_FALSE, &MVP[0][0]);
			glUniform3fv(1, 1, &lightDir[0]);
			glUniform3fv(2, 1, &eyePos[0]);
			glDrawArrays(GL_TRIANGLES, 0, scene->triangles.size() * 3);

			glUseProgram(lineProgram);
			glBindVertexArray(lineVAO);
			glUniformMatrix4fv(0, 1, GL_FALSE, &MVP[0][0]);
			glDrawArrays(GL_LINES, 0, scene->bvh->nodes.size() * 24);

			glfwSwapBuffers(window);
		}
		glfwDestroyWindow(window);
		glfwTerminate();
		clear();
		exit(0);
	}

	void clear()
	{
		glDeleteProgram(phoneProgram);
		glDeleteProgram(lineProgram);
		glDeleteVertexArrays(1, &modelVAO);
		glDeleteVertexArrays(1, &lineVAO);
		glDeleteBuffers(1, &modelVerts);
		glDeleteBuffers(1, &modelNormals);
		glDeleteBuffers(1, &lineVerts);
	}

private:
	void initWindow()
	{
		if (!glfwInit())
		{
			exit(EXIT_FAILURE);
		}

		window = glfwCreateWindow(scene->state.camera.resolution.x,
			scene->state.camera.resolution.y, "BVH Debug Window", NULL, NULL);
		if (!window)
		{
			glfwTerminate();
			throw std::exception();
		}
		glfwMakeContextCurrent(window);
		glfwSetKeyCallback(window, keyCallback);
		glfwSetCursorPosCallback(window, mousePositionCallback);
		glfwSetMouseButtonCallback(window, mouseButtonCallback);

		// Set up GL context
		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK)
		{
			throw std::exception();
		}
	}

	void initShaders()
	{
		phoneProgram = glslUtility::createProgramFromSrc(phoneVsSrc, phonePsSrc);
		lineProgram = glslUtility::createProgramFromSrc(lineVsSrc, linePsSrc);
	}

	void initVertexBuffers()
	{
		std::vector<glm::vec3> modelPos;
		std::vector<glm::vec3> modelNrm;

		for (const auto &tri : scene->triangles)
		{
			glm::mat4 T = scene->geoms[tri.meshIdx].transform;
			modelPos.emplace_back(glm::vec3(T * glm::vec4(tri.positions[0], 1.f)));
			modelPos.emplace_back(glm::vec3(T * glm::vec4(tri.positions[1], 1.f)));
			modelPos.emplace_back(glm::vec3(T * glm::vec4(tri.positions[2], 1.f)));
			modelNrm.emplace_back(glm::vec3(T * glm::vec4(tri.normals[0], 0.f)));
			modelNrm.emplace_back(glm::vec3(T * glm::vec4(tri.normals[1], 0.f)));
			modelNrm.emplace_back(glm::vec3(T * glm::vec4(tri.normals[2], 0.f)));
		}

		std::vector<glm::vec3> linePos;

		auto addBBoxVerts = [&linePos](const BVH::BBox &b)
		{
			glm::vec3 corners[8] =
			{
				{ b.p_min.x, b.p_min.y, b.p_min.z },
				{ b.p_max.x, b.p_min.y, b.p_min.z },
				{ b.p_min.x, b.p_max.y, b.p_min.z },
				{ b.p_min.x, b.p_min.y, b.p_max.z },
				{ b.p_max.x, b.p_max.y, b.p_min.z },
				{ b.p_max.x, b.p_min.y, b.p_max.z },
				{ b.p_min.x, b.p_max.y, b.p_max.z },
				{ b.p_max.x, b.p_max.y, b.p_max.z },
			};

			linePos.emplace_back(corners[0]);
			linePos.emplace_back(corners[1]);
			linePos.emplace_back(corners[1]);
			linePos.emplace_back(corners[4]);
			linePos.emplace_back(corners[4]);
			linePos.emplace_back(corners[2]);
			linePos.emplace_back(corners[2]);
			linePos.emplace_back(corners[0]);

			linePos.emplace_back(corners[0]);
			linePos.emplace_back(corners[2]);
			linePos.emplace_back(corners[1]);
			linePos.emplace_back(corners[5]);
			linePos.emplace_back(corners[4]);
			linePos.emplace_back(corners[7]);
			linePos.emplace_back(corners[2]);
			linePos.emplace_back(corners[6]);

			linePos.emplace_back(corners[3]);
			linePos.emplace_back(corners[5]);
			linePos.emplace_back(corners[5]);
			linePos.emplace_back(corners[7]);
			linePos.emplace_back(corners[7]);
			linePos.emplace_back(corners[6]);
			linePos.emplace_back(corners[6]);
			linePos.emplace_back(corners[3]);
		};

		for (const auto &node : scene->bvh->nodes)
		{
			addBBoxVerts(node.bounds);
		}
		
		glGenVertexArrays(1, &modelVAO);
		glBindVertexArray(modelVAO);
		glGenBuffers(1, &modelVerts);
		glBindBuffer(GL_ARRAY_BUFFER, modelVerts);
		glBufferData(GL_ARRAY_BUFFER, modelPos.size() * sizeof(glm::vec3), modelPos.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glGenBuffers(1, &modelNormals);
		glBindBuffer(GL_ARRAY_BUFFER, modelNormals);
		glBufferData(GL_ARRAY_BUFFER, modelNrm.size() * sizeof(glm::vec3), modelNrm.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glGenVertexArrays(1, &lineVAO);
		glBindVertexArray(lineVAO);
		glGenBuffers(1, &lineVerts);
		glBindBuffer(GL_ARRAY_BUFFER, lineVerts);
		glBufferData(GL_ARRAY_BUFFER, linePos.size() * sizeof(glm::vec3), linePos.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindVertexArray(0);
	}

	GLuint modelVerts, modelNormals, modelVAO;
	GLuint lineVerts, lineVAO;

	GLuint phoneProgram, lineProgram;
	
	GLFWwindow *window;
	Scene *scene;
};


//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

	//BVHDebug bvhdebug(scene);
	//bvhdebug.run();

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec4 pix4 = renderState->image[index];
			glm::vec3 pix(pix4.x / pix4.w, pix4.y / pix4.w, pix4.z / pix4.w);
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
		numPathsFinished = 0.0;
		totalElapsedTimeMs = 0.0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
        pathtrace(pbo_dptr, frame, iteration);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&iterationTimeMs, start, stop);
		totalElapsedTimeMs += iterationTimeMs;

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
		++iteration;
    } else {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
