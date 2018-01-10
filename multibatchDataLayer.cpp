//#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multibatchDataLayer.hpp"
//#include "caffe/layers/base_data_layer.hpp"
//#include "caffe/layers/multibatchDataLayer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//#define DEBUGMULTIBATCH

using namespace std;
namespace caffe {

template <typename Dtype>
MultibatchDataLayer<Dtype>::~MultibatchDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultibatchDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.multi_batch_data_param().new_height();
  const int new_width = this->layer_param_.multi_batch_data_param().new_width();
  const bool is_color = this->layer_param_.multi_batch_data_param().is_color();
  string root_folder = this->layer_param_.multi_batch_data_param().root_folder();
  this->rand_gray = this->layer_param_.multi_batch_data_param().rand_gray();
  this->rand_identity = this->layer_param_.multi_batch_data_param().rand_identity();
  this->_identityNumPerBatch = this->layer_param_.multi_batch_data_param().identity_num_per_batch();
  this->_imgNumPerIdentity = this->layer_param_.multi_batch_data_param().img_num_per_identity();
  CHECK(this->_identityNumPerBatch *_imgNumPerIdentity == this->layer_param_.multi_batch_data_param().batch_size())
	  <<"identitynum^2 should equal to batchsize";

  //CHECK(this->layer_param_.image_data_param().shuffle() == false) << "MultibatchDataLayer dont need shuffle";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multi_batch_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  int img_num = 0;
  while (infile >> filename >> label) {
	  _randimg_lines_.push_back(std::make_pair(filename, label));
	  this->_instanceStrorage_[label].second.push_back(filename);
    this->_identityIDs.push_back(label);
	  img_num++;
  }
  //iterate the map and initialize each label's list space
  //typedef map<int, std::pair<pair<int, bool>, std::vector<string> > >::iterator it_type;
  //for (it_type iter = this->_instanceStrorage_.begin(); iter != this->_instanceStrorage_.end(); iter++){
  for (map<int, std::pair<pair<int, bool>, std::vector<string> > >::iterator iter = this->_instanceStrorage_.begin(); iter != this->_instanceStrorage_.end(); iter++){
	  //iter->second.second = vector<int>(iter->second.first.first);//allocate the memory space
	  //iter->second.second.clear();//recount, but the space still exists.
	  iter->second.first.first = 0;//pair.first.first reset to 0
	  iter->second.first.second = false;
  }

  LOG(INFO) << "A total of " << img_num << " images. and "<< this->_instanceStrorage_.size() <<" identities.";

  //int rand_lbl_key = _instanceStrorage_.keys()[0];
  //LOG(INFO) << "FIRST IMG PATH: "<<_instanceStrorage_[].second[0];
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + _instanceStrorage_[label].second[0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << _instanceStrorage_[label].second[0];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.multi_batch_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;

  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);

  /* old multi-batch data
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  */
}

// map<int, std::pair<pair<int,bool>,std::vector<int> > > _instanceStrorage_;
template <typename Dtype>
void MultibatchDataLayer<Dtype>::ShuffleImages() {
	//Reset container
	//typedef std::map < int, pair<int, vector<int>>>::iterator it_type;
	//for (it_type iter = this->_instanceStrorage_.begin(); iter != this->_instanceStrorage_.end(); iter++){
	//	iter->second.second.clear();//recount, but the space exists.
	//	iter->second.first.first = 0;//pair.first.first reset to 0
	//	iter->second.first.second = false;//set looped sign to 0
	//}
	//this->_looped_count = 0;
  //caffe::rng_t* prefetch_rng =
  //    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //shuffle(lines_.begin(), lines_.end(), prefetch_rng);

  /*for (int i = 0; i < lines_.size();i++){
	  this->_instanceStrorage_[lines_[i].second].second.push_back(i);
	  
  }*/
  

}

template <typename Dtype>
void MultibatchDataLayer<Dtype>::gen_rand_identity(Dtype* identityid){
	caffe_rng_uniform<Dtype>(1, (Dtype)0, (Dtype)(this->_identityIDs.size() - 1), identityid);
  *identityid = _identityIDs[*identityid];
}

template <typename Dtype>
 void MultibatchDataLayer<Dtype>::gen_rand_image(Dtype* identityid){
	Dtype imageidx;
	caffe_rng_uniform<Dtype>(1, (Dtype)0, (Dtype)(this->_randimg_lines_.size() - 1), &imageidx);
	*identityid = this->_randimg_lines_[(int)imageidx].second;
}

template <typename Dtype>
void MultibatchDataLayer<Dtype>::InternalThreadEntry(){
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  //CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  MultiBatchDataParameter multi_batch_data_param = this->layer_param_.multi_batch_data_param();
  const int batch_size = multi_batch_data_param.batch_size();
  const int new_height = multi_batch_data_param.new_height();
  const int new_width = multi_batch_data_param.new_width();
  const bool is_color = multi_batch_data_param.is_color();
  string root_folder = multi_batch_data_param.root_folder();

  int t_label = _instanceStrorage_.begin()->first;
  cv::Mat cv_img = ReadImageToCVMat(root_folder + _instanceStrorage_[t_label].second[0], new_height, new_width, is_color);



  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
//LOG(INFO)<<"LOG1";
  for (int batchIdentIdx = 0; batchIdentIdx < this->_identityNumPerBatch; batchIdentIdx++){
    Dtype temp_IdentLabel;
    //caffe_rng_uniform<Dtype>(1, (Dtype)0, (Dtype)(_instanceStrorage_.size() - 1), &temp_IdentLabel);
    if (this->rand_identity){
      this->gen_rand_identity(&temp_IdentLabel);
    }
    else{
      this->gen_rand_image(&temp_IdentLabel);
    }
      //LOG(INFO)<<"LOG11";
    int thisIdentLabel = (int)temp_IdentLabel;
    for (int identImgIdx = 0; identImgIdx < this->_imgNumPerIdentity; identImgIdx++){
      //LOG(INFO)<<"LOG_START";
      int item_id = batchIdentIdx * this->_imgNumPerIdentity + identImgIdx;
      timer.Start();
      Dtype tempThisImgIdxInList;
      //LOG(INFO)<<"LOG10_1";
      caffe_rng_uniform<Dtype>(1, (Dtype)0, (Dtype)(_instanceStrorage_[thisIdentLabel].second.size()), &tempThisImgIdxInList);
      if(tempThisImgIdxInList>= _instanceStrorage_[thisIdentLabel].second.size()){
        tempThisImgIdxInList = tempThisImgIdxInList -1;
      }
      int thisImgIdxInList = (int)tempThisImgIdxInList;
      //LOG(INFO)<<"FILELIST SIZE"<<_instanceStrorage_[thisIdentLabel].second.size();
      //LOG(INFO)<<"thisImgIdxInList="<<thisImgIdxInList;
      //LOG(INFO)<<"LOG10_2";
      //cv::Mat cv_img = ReadImageToCVMat(root_folder + _instanceStrorage_[thisIdentLabel].second[thisImgIdxInList],
      //  new_height, new_width, is_color);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + _instanceStrorage_[thisIdentLabel].second[thisImgIdxInList],  new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << _instanceStrorage_[thisIdentLabel].second[thisImgIdxInList];
      //LOG(INFO)<<"LOG11";
      read_time += timer.MicroSeconds();
      timer.Start();
      int offset = this->prefetch_data_.offset(item_id);
      //LOG(INFO)<<"LOG12";
      // don't clear for it
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      //LOG(INFO)<<"LOG13";
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      //LOG(INFO)<<"LOG14";
      trans_time += timer.MicroSeconds();
      prefetch_label[item_id] = thisIdentLabel;
      //LOG(INFO)<<"LOG_END";
       
#ifdef DEBUGMULTIBATCH
      LOG(INFO) << "path:" << _instanceStrorage_[thisIdentLabel].second[thisImgIdxInList] <<"  label:" << thisIdentLabel;
#endif
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}


INSTANTIATE_CLASS(MultibatchDataLayer);
REGISTER_LAYER_CLASS(MultibatchData);

}  // namespace caffe
//#endif  // USE_OPENCV
