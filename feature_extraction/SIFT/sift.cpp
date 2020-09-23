/**
 * opencv 第一个程序
 * 涉及到图像拉伸、旋转、sift 算子匹配
 * @time: 2020/03/24
 * @author: 刘畅
**/

// STL 标准库
#include <vector>
#include <assert.h>
#include <iostream>
#include <exception>  
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>


class engine {
	using Image = cv::Mat;
	using points = std::vector<cv::KeyPoint>;
public:
	// 析构函数, 清理窗口等
	~engine() noexcept {
		try {
			cv::destroyAllWindows();
			std::cout << "\nall of the Windows has been closed !\n";
		} catch(std::exception& e) { std::cout << "exception occurred when closing all Windows !\n"; }
	}

	// 窗口根据信号 signal 显示
	void waitKey(const int signal=0) {
		cv::waitKey(signal);
	}

	// 根据文件名 name 读取图像
	Image imread(const std::string& name) {
		return cv::imread(name);
	}

	// 将图像 one 写入到文件 name
	bool imwrite(const std::string& name, const Image& one) {
		try {
			cv::imwrite(name, one);
		} catch(std::exception& e) { std::cout << e.what() << "\noccurred when saving image" << name << "\n"; return false; }
		std::cout << "the result is writen to ===> " << name << "\n";
		return true;
	}

	// 展示图像 image, 窗口名 name, 是否停留 wait
	void imshow(const Image& image, std::string name="YHL", const bool wait=true) {
		cv::imshow(name, image);
		if(wait) this->waitKey(0);
	}

	// 旋转图像 image, 角度 angle, 逆时针
	Image rotate(Image image, const double angle) {
		Image result;
		cv::Size result_size(image.cols, image.rows);
		// 找出图像中点
		cv::Point2f center(static_cast<float>(image.cols / 2.), static_cast<float>(image.rows / 2.));
		// 根据 angle 计算旋转矩阵
		Image rot_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		// 根据旋转矩阵进行仿射变换
		cv::warpAffine(image, result, rot_matrix, result_size, cv::INTER_LINEAR);
		return result;
	}

	// 图像拉伸缩小 image, 目标大小 (width, height), 插值方式 _interpolation
	Image resize(Image& image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		Image result;
		cv::resize(image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	// 更改图像 image 的通道
	Image cvtColor(const Image& image, const int mode=cv::COLOR_BGR2GRAY) {
		Image image_gray;
		cv::cvtColor(image, image_gray, mode);
		return image_gray;
	}

	// 检测图像 image 的 sift 关键点, 返回 <关键点, 标红的图像>
	std::pair<points, Image> sift_detect(const Image& image) {
		if(this->sift_detector == nullptr)
			this->sift_detector = cv::xfeatures2d::SiftFeatureDetector::create();
		assert(this->sift_detector);
		// 转化为灰度图
		auto image_gray = this->cvtColor(image);
		points key_points;
		// sift 检测, 结果放在 key_points
		this->sift_detector->detect(image_gray, key_points);
		Image result;
		// 将 key_points 在 image 上画出来, 结果放在 result 
		cv::drawKeypoints(image, key_points, result);
		return std::make_pair(key_points, result);
	}

	// 根据图像 image 的 sift 关键点, 提取 sift 描述符
	Image get_sift_descriptions(const Image& image, points& key_points) {
		if(this->sift_extractor == nullptr)
			this->sift_extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
		assert(this->sift_extractor);
		Image description;
		this->sift_extractor->compute(this->cvtColor(image), key_points, description);
		return description;
	}

	// 根据图像 lhs 和 rhs 的关键点 key_points 和 描述符 descriptions 画出匹配图
	Image match_descriptions(
			const Image& lhs, const points& lhs_key_points, const Image& lhs_descriptions, 
			const Image& rhs, const points& rhs_key_points, const Image& rhs_descriptions, 
			const double rate=0.1) {
		// 清空上次的操作
		if(not this->matcher.empty()) 
			this->matcher.clear();
		// 放入描述符
		this->matcher.add(std::vector<Image>(1, lhs_descriptions));
		this->matcher.train();
		// 设定匹配点矩阵
		std::vector< std::vector<cv::DMatch> > matches;
		this->matcher.knnMatch(rhs_descriptions, matches, 2);
		// good_matches 存储那些符合标准的点
		std::vector<cv::DMatch> good_matches;
		for(const auto it : matches) 
			if(it[0].distance < rate * it[1].distance)
				good_matches.emplace_back(it[0]);
		// 顺序......
		Image match_result;
		cv::drawMatches(rhs, rhs_key_points, lhs, lhs_key_points, good_matches, match_result);
		return match_result;
	}

private:
	// 匹配算法, 选 bf
	cv::BFMatcher matcher;
	// sift 检测算子
	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift_detector = nullptr;
	// sift 提取描述符的算子
	cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> sift_extractor = nullptr;
};


int main() {
	// 声明 module cv2, 模仿 python 版本
	engine cv2;

	// 读入高分辨率图像
	auto high_res_image = cv2.imread("./a4644_label.jpg"); 

	// 缩小得到低分辨率图像和旋转的图像, 为后续匹配做准备
	auto low_res_image = cv2.resize(high_res_image, 400, 600);
	auto low_res_image_rotated = cv2.rotate(low_res_image, 20);

	// 分别应用 sift 算法检测两张图象的关键点
	auto train_result = cv2.sift_detect(low_res_image);
	auto test_result = cv2.sift_detect(low_res_image_rotated);

	// 分别提取两张图的 sift 描述符
	auto train_description = cv2.get_sift_descriptions(low_res_image, train_result.first);
	auto test_description = cv2.get_sift_descriptions(low_res_image_rotated, test_result.first);
	
	// 得到匹配图
	auto match_result = cv2.match_descriptions(
		low_res_image, train_result.first, train_description, 
		low_res_image_rotated, test_result.first, test_description, 
		0.1);
	cv2.imshow(match_result);
	cv2.imwrite("./match_result.jpg", match_result);

	return 0;
}