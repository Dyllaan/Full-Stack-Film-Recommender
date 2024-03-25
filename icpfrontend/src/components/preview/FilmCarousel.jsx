// Import Swiper React components
import { Swiper, SwiperSlide } from 'swiper/react';

// Import Swiper styles
import 'swiper/css';
import 'swiper/css/free-mode';
import 'swiper/css/pagination';

import './styles.css';

// import required modules
import { FreeMode, Pagination } from 'swiper/modules';

import PropTypes from 'prop-types';
import MyDrawer from '../Drawer';

const FilmCarousel = ({ posters }) => {

    const url= `https://image.tmdb.org/t/p/w500`;
    
  return (
    <>
      <Swiper
        slidesPerView={4}
        spaceBetween={30}
        freeMode={true}
        grabCursor={true}
        loop={true}
        pagination={{
          clickable: true,
        }}
        modules={[FreeMode, Pagination]}
        className="mySwiper"
      >
        <div className="m-4">
        {posters.map((poster, index) => (
            <SwiperSlide key={index} className='hover-scale'>
              <img src={url+poster.poster_path}>
                </img></SwiperSlide>
        ))}
        </div>
      </Swiper>
      <MyDrawer />
    </>
  );
};

FilmCarousel.propTypes = {
    posters: PropTypes.array.isRequired,
    };

export default FilmCarousel;
