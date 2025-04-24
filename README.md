<!-- PROJECT SHIELDS -->
<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/wanghley)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wanghley/neuromappr">
    <img src="image.gif" alt="Logo" width="280">
  </a>

  <h3 align="center">Neuromappr</h3>

  <p align="center">
    A Brain-Computer Interface (BCI) movement decoder using Support Vector Machines on high-dimensional EEG signals.
    <br />
    <a href="#"><strong>Explore the code »</strong></a>
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**Neuromappr** is a machine learning project that applies Support Vector Machines (SVMs) to decode imagined and overt hand movements from EEG signals in a brain-computer interface (BCI) setting. This project was developed for the *ECE 580: Introduction to Machine Learning* course and showcases the application of classifiers in high-dimensional data environments.

The dataset consists of 204 features from 102 electrodes (x and y field gradients) across trials involving either imagined or actual left/right hand movements. The goal is to determine the user's intended direction—left or right—using classification techniques with proper model validation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<img src="https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white" alt="matlab"> <img src="https://img.shields.io/badge/SVM-00599C?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="svm"> <img src="https://img.shields.io/badge/EEG%20Data-ffb347?style=for-the-badge" alt="eeg">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- MATLAB (R2020+ recommended)
- Signal Processing Toolbox
- Data files (available on course platform)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/wanghley/neuromappr.git
   cd neuromappr
   ```

2. Open the project folder in MATLAB and run:
   ```matlab
   main.m
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

The project uses a two-level cross-validation approach to train and evaluate SVM classifiers on both imagined and overt EEG movement datasets. It includes:

- Data preprocessing and organization by class (left/right)
- Implementation of a linear baseline SVM and comparison with other kernels
- Optimization of the regularization parameter α
- Visualization of decision function weights over electrode positions

Each trial is represented by a 204-element feature vector, and classification is performed using stratified k-fold (6-fold outer, 5-fold inner) cross-validation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Load and process EEG datasets
- [x] Implement linear SVM for movement classification
- [x] Optimize regularization parameter
- [x] Visualize electrode weights
- [ ] Implement kernel-based SVM models
- [ ] Integrate phase/magnitude transformation
- [ ] Enhance UI for real-time testing

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions make the open source community such an amazing place to learn and grow. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Wanghley Soares Martins - [@wanghley](https://instagram.com/wanghley) - wanghley@wanghley.com

Project Link: [https://github.com/wanghley/neuromappr](https://github.com/wanghley/neuromappr)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Dr. Stacy Tantum – Course Instructor, ECE 580
* [IEEE BCI Research](https://ieeexplore.ieee.org/document/2407272)
* MATLAB Documentation
* Colleagues and classmates at Duke

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/wanghley/neuromappr?style=for-the-badge
[contributors-url]: https://github.com/wanghley/neuromappr/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wanghley/neuromappr.svg?style=for-the-badge
[forks-url]: https://github.com/wanghley/neuromappr/network/members
[stars-shield]: https://img.shields.io/github/stars/wanghley/neuromappr.svg?style=for-the-badge
[stars-url]: https://github.com/wanghley/neuromappr/stargazers
[issues-shield]: https://img.shields.io/github/issues/wanghley/neuromappr.svg?style=for-the-badge
[issues-url]: https://github.com/wanghley/neuromappr/issues
[license-shield]: https://img.shields.io/github/license/wanghley/neuromappr.svg?style=for-the-badge
[license-url]: https://github.com/wanghley/neuromappr/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/wanghley
