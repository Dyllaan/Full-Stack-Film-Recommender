import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/free-mode';
import 'swiper/css/pagination';
import { useState } from 'react';
import '../styles.css';
import { FreeMode, Pagination } from 'swiper/modules';
import { Box } from '@mui/material';

// PropTypes import is correct; no change needed here
import PropTypes from 'prop-types';

export default function TMDbImages({ images }) {

  const [selectedItem, setSelectedItem] = useState(images[0]?.file_path || null);

  if(!images) {
    return;
  }

  const slicedImages = images.slice(0, 10);

  return (
    <div className="flex flex-col gap-2">
      {/* Check if selectedItem is not null before trying to display the image */}
      <Box className="min-w-[30vw]">
        {selectedItem && (
          <img className="mx-auto m-2 object-scale-down rounded-lg" src={`https://image.tmdb.org/t/p/w1280${selectedItem}`} alt={`Movie Poster`} />
        )}
      </Box>
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
          480: {
            slidesPerView: 3,
            spaceBetween: 20,
          },
          768: {
            slidesPerView: 5,
            spaceBetween: 20,
          },
          1024: {
            slidesPerView: 6,
            spaceBetween: 20,
          },
        }}
        style={{
          "--swiper-pagination-color": "#631956",
          "--swiper-pagination-bullet-inactive-color": "#999999",
          "--swiper-pagination-bullet-inactive-opacity": "1",
          "--swiper-pagination-bullet-size": "12px",
          "--swiper-pagination-bullet-horizontal-gap": "6px",
        }}
      >
        {slicedImages.map((image, index) => (
          <SwiperSlide key={index} onClick={() => setSelectedItem(image.file_path)}>
            <img src={`https://image.tmdb.org/t/p/w500${image.file_path}`} alt={`Movie Poster`} />
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
}

TMDbImages.propTypes = {
  images: PropTypes.array.isRequired,
};
