
import PropTypes from 'prop-types';

const FilmPosterGrid = ({ films }) => {
    const url = `https://image.tmdb.org/t/p/w200`;

    return (
        <div className="grid grid-cols-5 gap-4 w-[90vw] mx-auto">
          {films.map((film, index) => (
            <div key={index} className="flex justify-center items-center">
                <img src={url + film.poster_path} alt={film.title} />
            </div>
          ))}
        </div>
      );
};

FilmPosterGrid.propTypes = {
    films: PropTypes.array.isRequired,
};

export default FilmPosterGrid;
