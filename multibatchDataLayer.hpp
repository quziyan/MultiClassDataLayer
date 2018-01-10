#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "pipecall.h"
namespace caffe {
/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultibatchDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
   explicit MultibatchDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
   virtual ~MultibatchDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultibatchData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  /*
  //Obsolate: In this version, this function is too advancing.
  virtual void load_batch(Batch<Dtype>* batch);
  */
  
  //in this layer, these variable just save the information for generating by random image.
  vector<std::pair<std::string, int> > _randimg_lines_;
  vector<int> _identityIDs;
  //in map, the key corresponds label
  //in map's pair, the FIRST is used for counting image number in shuffle stage., and being the cursor in filling batch stage.
  //bool indicate whether it has already been loop once.
  //in map's pair, the SECOND indicate the index list in the shuffled lines.
  map<int, std::pair<pair<int,bool>,std::vector<string> > > _instanceStrorage_;
  //for each label's data, if it looped once, this variable minus 1.
  //when all label has been looped once, then shuffle again.
  //int _looped_count;
  //int lines_id_;

  int _identityNumPerBatch,_imgNumPerIdentity;
  bool rand_gray;
  //whether it generate batch by rand-identity or rand-image
  bool rand_identity;
  void gen_rand_identity(Dtype* identityid);
  void gen_rand_image(Dtype* identityid);
};


}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
