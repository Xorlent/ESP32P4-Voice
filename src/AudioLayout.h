#pragma once

#include <cstddef>
#include <cstdint>

namespace cactus {
class CactArchive;
}

// Location of the Whistle audio tensors inside a .cact archive.
//
// The archive is nameless; the audio region follows the text model in a fixed
// order:
//   audio.manifest (13 fp32)
//   per decoder layer: cross_norm, q/k/v/gate/out projections, q_norm, k_norm,
//                      post_cross_norm, cross_gate            (10 each)
//   pe_gate
//   per encoder layer: 6 norms, 5 projections, q_norm, k_norm, attn_gate,
//                      pw1, pw2, dw, hadamard_mlp_0 (13), hadamard_mlp (13)
//                                                              (43 each)
//   encoder mHC: a_pre, a_post, a_res, b_pre, b_post, b_res,
//                phi_pre, phi_post, phi_res, final_norm          (10)
//   stem: w, dw_1, pw_1, dw_2, pw_2, out                         (6)
//   mel filterbank                                                (1)
//   tokenizer                                                     (1)
//
// All indices are directory positions. `findAudioLayout` finds the manifest by
// its magic and validates every shape before returning true.
namespace whistle {

struct AudioLayout {
  bool present = false;

  uint32_t manifest = 0;
  uint32_t cross_base = 0;      // first cross-attention tensor (layer 0)
  uint32_t pe_gate = 0;
  uint32_t enc_base = 0;        // first encoder-layer tensor (layer 0)
  uint32_t enc_mhc_base = 0;    // encoder mHC a_pre
  uint32_t enc_final_norm = 0;

  uint32_t stem_w = 0;
  uint32_t stem_dw1 = 0;
  uint32_t stem_pw1 = 0;
  uint32_t stem_dw2 = 0;
  uint32_t stem_pw2 = 0;
  uint32_t stem_out = 0;
  uint32_t mel = 0;
  uint32_t tokenizer = 0;

  uint32_t d_model = 0;
  uint32_t enc_layers = 0;
  uint32_t enc_lanes = 0;
  uint32_t dec_layers = 0;
  uint32_t dec_lanes = 0;
  uint32_t conv_kernel = 0;
  uint32_t stem = 0;
  uint32_t stem_stages = 0;
  uint32_t mels = 0;
  uint32_t rate = 0;
  uint32_t fft = 0;
  uint32_t win = 0;
  uint32_t hop = 0;

  static constexpr uint32_t kCrossPerLayer = 10;
  static constexpr uint32_t kEncPerLayer = 43;
  static constexpr uint32_t kMhcCount = 9;

  uint32_t crossIndex(uint32_t layer, uint32_t rel) const {
    return cross_base + layer * kCrossPerLayer + rel;
  }
  uint32_t encIndex(uint32_t layer, uint32_t rel) const {
    return enc_base + layer * kEncPerLayer + rel;
  }
};

// Returns false when the archive has no audio region (for example needle3.cact)
// or when its layout does not match the expected export order.
bool findAudioLayout(const cactus::CactArchive& archive, AudioLayout* out);

}  // namespace whistle
