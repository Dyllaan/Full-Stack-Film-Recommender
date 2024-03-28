import PropTypes from 'prop-types';
import FilmThumbnailCard from '../FilmThumbnailCard';
import Grid from '@mui/material/Grid';

function RatedFilms(props) {
    const { ratedFilms } = props;
    return (
        <div className="m-2 p-2 bg-secondary-black rounded-lg">
            <Grid container spacing={1}>
                {ratedFilms.length > 0 && ratedFilms.map((film, index) => (
                    <Grid item key={index}>
                    <div className="max-w-12 hover-scale" key={index}>
                        <FilmThumbnailCard film={film} />
                    </div>
                    </Grid>
                ))}
            </Grid>
      </div>
    )
}

RatedFilms.propTypes = {
    ratedFilms: PropTypes.array.isRequired,
}

export default RatedFilms;