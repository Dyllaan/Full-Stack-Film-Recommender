// Import Swiper React components
import { Swiper, SwiperSlide } from 'swiper/react';

// Import Swiper styles
import 'swiper/css';
import 'swiper/css/free-mode';
import 'swiper/css/pagination';
import { useState } from 'react';

import './styles.css';

// import required modules
import { FreeMode, Pagination } from 'swiper/modules';

import PropTypes from 'prop-types';
import MyDrawer from '../Drawer';
import RateFilmCard from '../films/RateFilmCard';

const FilmCarousel = ({ films }) => {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedSlide, setSelectedSlide] = useState(null); // Track the selected slide

  // Function to open the drawer with specific slide data
  const handleSlideClick = (index) => {
    setSelectedSlide(index); // Store the index or any identifier of the slide
    setIsDrawerOpen(true); // Open the drawer
  };

  const url = `https://image.tmdb.org/t/p/w500`;

  if(!films) return;
  if(films.length === 0) return;

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
        className="swiper p-4"
      >
        {films && films.length > 0 && films.map((poster, index) => (
          <SwiperSlide key={index} className='hover-scale' onClick={() => handleSlideClick(index)}>
            {poster.poster_path ? (
              <div>
                <RateFilmCard film={poster} />
              </div>
            ) : (
              <div className="text-center bg-secondary-black">
                <h2 className="text-2xl">No Poster Available</h2>
              </div>
            )}

          </SwiperSlide>
        ))}
      </Swiper>
      <MyDrawer isOpen={isDrawerOpen} setIsOpen={setIsDrawerOpen} selectedSlide={selectedSlide} films={films} />
    </>
  );
};


FilmCarousel.propTypes = {
  films: PropTypes.array.isRequired,
};

export default FilmCarousel;
