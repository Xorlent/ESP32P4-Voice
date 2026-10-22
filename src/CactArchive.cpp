#include "CactArchive.h"

namespace cactus {

namespace {

// Each directory entry is exactly 44 bytes.
#pragma pack(push, 1)
struct DirEntry {
  uint8_t dtype;
  uint8_t ndim;
  uint16_t pad;
  uint32_t shape[4];
  uint64_t offset;
  uint64_t nbytes;
  uint32_t group;
  uint32_t bits;
};
#pragma pack(pop)
static_assert(sizeof(DirEntry) == 44, "directory entry must be 44 bytes");

}  // namespace

bool CactArchive::parse(const uint8_t* data, size_t size) {
  parsed_ = false;
  num_tensors_ = 0;
  codebooks_ = nullptr;
  tensors_.clear();
  if (data == nullptr || size < 196) {
    return false;
  }

  base_ = data;
  size_ = size;

  // 196-byte header: 49 x u32, little-endian.
  const uint32_t* h = reinterpret_cast<const uint32_t*>(data);
  header_.tag = h[0];
  header_.num_tensors = h[1];
  header_.codebook_len = h[2];
  for (size_t i = 0; i < 49; ++i) {
    header_.f[i] = h[i];
  }

  if (header_.tag != kNeedle3Tag && header_.tag != kNeedle2Tag) {
    return false;
  }
  if (header_.num_tensors > kMaxTensors) {
    return false;
  }

  // Codebooks follow the header immediately.
  codebooks_ = reinterpret_cast<const float*>(data + 196);

  // The nameless directory follows the codebooks (44 bytes per tensor).
  const uint8_t* dir = data + 196 + static_cast<size_t>(header_.codebook_len) * 4;
  tensors_.resize(header_.num_tensors);
  for (size_t i = 0; i < header_.num_tensors; ++i) {
    const DirEntry* e = reinterpret_cast<const DirEntry*>(dir + i * 44);
    if (e->offset + e->nbytes > size || e->offset < 196) {
      return false;  // offset/nbytes out of range
    }
    TensorMeta& m = tensors_[i];
    m.dtype = static_cast<DType>(e->dtype);
    m.ndim = e->ndim;
    for (size_t j = 0; j < 4; ++j) {
      m.shape[j] = e->shape[j];
    }
    m.offset = e->offset;
    m.nbytes = e->nbytes;
    m.group = e->group;
    m.bits = e->bits;
  }
  num_tensors_ = header_.num_tensors;
  parsed_ = true;
  return true;
}

const TensorMeta* CactArchive::tensor(size_t i) const {
  return (i < num_tensors_) ? &tensors_[i] : nullptr;
}

}  // namespace cactus
