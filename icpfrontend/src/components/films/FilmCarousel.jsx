import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/free-mode';
import 'swiper/css/pagination';
import { useState } from 'react';
import './styles.css';
import { FreeMode, Pagination } from 'swiper/modules';

import PropTypes from 'prop-types';
import FilmDrawer from './FilmDrawer';
import Loading from '../Loading';

/**
  * Styling credit:
  * https://stackoverflow.com/questions/65590148/swiperjs-how-do-you-style-the-pagination-bullets
  * User: Hotcaffe, 9 July 2022
  */

const FilmCarousel = ({ films, loading, title }) => {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedSlide, setSelectedSlide] = useState(null);

  if(loading) {
    return <Loading loading={loading} />
  }

  const handleSlideClick = (index) => {
    setSelectedSlide(index);
    setIsDrawerOpen(true);
  };

  const url = `https://image.tmdb.org/t/p/w500`;

  return (
    <div className="w-[75vw] flex flex-col gap-2">
      <h1 className="truncate">{title}</h1><div>
        <Swiper
          slidesPerView={1}
          spaceBetween={30}
          freeMode={true}
          grabCursor={true}
          loop={true}
          pagination={{
            clickable: true,
          }}
          modules={[FreeMode, Pagination]}
          className=""
          breakpoints={{
            240: {
              slidesPerView: 2,
              spaceBetween: 10,
            },
            // <= 480ox
            480: {
              slidesPerView: 3,
              spaceBetween: 20,
            },
            // <= 768px
            768: {
              slidesPerView: 5,
              spaceBetween: 20,
            },
            // <= 1024px
            1024: {
              slidesPerView: 6,
              spaceBetween: 20,
            },
          }}
          style={{
            "--swiper-pagination-color": "#631956",
            "--swiper-pagination-bullet-inactive-color": "#999999",
            "--swiper-pagination-bullet-inactive-opacity": "1",
            "--swiper-pagination-bullet-size": "8px",
            "--swiper-pagination-bullet-horizontal-gap": "6px"
          }}
        >
          {films.map((poster, index) => (
            <SwiperSlide key={index} className='hover-scale' onClick={() => handleSlideClick(index)}>
              <img src={url + poster.poster_path} alt={`Movie Poster ${index}`} />
            </SwiperSlide>
          ))}
          </Swiper>
          <FilmDrawer isOpen={isDrawerOpen} setIsOpen={setIsDrawerOpen} selectedSlide={selectedSlide} posters={films} />
      </div>
    </div>
  );
};


FilmCarousel.propTypes = {
  films: PropTypes.array.isRequired,
  loading: PropTypes.bool.isRequired,
  title: PropTypes.string.isRequired,
};

export default FilmCarousel;
