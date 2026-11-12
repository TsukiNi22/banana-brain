/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 12/11/2026 by @author Tsukini

File Name:
##  @file Metrics.hpp

File Description:
##  Metrics of a network on a dataset: classification (confusion, accuracy, precision, recall, f1) and losses
\**************************************************************/

#ifndef METRICS_H
    #define METRICS_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>     // _cold, _nodiscard
    #include "NetworkType.hpp"     // brain::UIType, brain::FType, brain::Values
    #include <filesystem>          // std::filesystem::path
    #include <ostream>             // std::ostream
    #include <vector>              // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// Filled by brain::NeuralNetwork::metrics (the class of a sample = argmax of its values)
class Metrics {
    private:
        /* classification */
        std::vector<std::vector<brain::UIType>> _confusion; // [expected][predicted]

        /* loss */
        brain::Values _losses; // mean loss of each output
        brain::FType _loss = 0.0; // global mean

    public:
        // ---------- Pre-Function -------- //
        /* classification */
        _nodiscard brain::FType accuracy(void) const; // share of the samples well classified
        _nodiscard brain::FType accuracy(const brain::UIType expected) const; // of one expected class
        _nodiscard brain::FType precision(const brain::UIType c) const; // share of the predicted c that are c
        _nodiscard brain::FType recall(const brain::UIType c) const; // share of the c predicted as c
        _nodiscard brain::FType f1(const brain::UIType c) const; // harmonic mean of the precision and the recall

        /* output */
        void csv(const std::filesystem::path& path) const; // the confusion matrix and the losses

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline const std::vector<std::vector<brain::UIType>>& confusion(void) const {return this->_confusion;};
        _cold _nodiscard inline const brain::Values& losses(void) const                              {return this->_losses;};
        _cold _nodiscard inline brain::FType loss(void) const                                        {return this->_loss;};

        // ------------ Operator ---------- //
        friend std::ostream& operator<<(std::ostream& os, const Metrics& metrics); // table

        Metrics& operator=(const Metrics& other) = default;
        Metrics& operator=(Metrics&& other) = default;

        // ---------- Constructor --------- //
        Metrics(const std::vector<std::vector<brain::UIType>>& confusion, const brain::Values& losses, const brain::FType loss): _confusion{confusion}, _losses{losses}, _loss{loss} { /* Nothing */ };

        Metrics() = default;
        Metrics(const Metrics& other) = default;
        Metrics(Metrics&& other) = default;

        // ----------- Destructor --------- //
        ~Metrics() = default;
};

} // namespace end
#endif /* METRICS_H */
