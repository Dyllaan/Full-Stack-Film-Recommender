import useFetchData from '../hooks/useFetchData';
import { useParams } from 'react-router-dom';
import useTMDb from '../hooks/useTMDb';
import { useEffect } from 'react';
import { Box, Paper, Typography, Container, Grid} from '@mui/material';
import FilmSidebar from '../components/films/page-components/FilmSidebar';
import TMDbImages from '../components/films/page-components/TMDbImages';
import fixMovieLensTitle from '../util/FixMovieLensTitle';
import FilmCarousel from '../components/films/FilmCarousel';
import RateFilm from '../components/films/RateFilm';
import useRating from '../hooks/useRating';

function FilmPage() {
  const { slug } = useParams();
  const { data : film, loading } = useFetchData(`movie/${slug}`);
  const { data, loading: tmdbLoading, doRun } = useTMDb(film.tmdb_id, false);
  const { data : tmdbImages, loading: tmdbImagesLoading, doRun: fetchImages } = useTMDb(film.tmdb_id, false, 'images');
  const { data: similarToX, loading: similarLoading, doRun: findSimilar } = useFetchData(`similar_movies/${slug}`, false);
  const { sendRating } = useRating();
  const [rating, setRating] = useState(0);

  useEffect(() => {
    if(film.tmdb_id) {
      doRun();
      fetchImages(); 
      findSimilar();
    }
  }, [film]);

  if(loading || tmdbLoading || tmdbImagesLoading || similarLoading) {
    return;
  }
  
  const releaseDate = (data.release_date ? new Date(data.release_date).toDateString() : "");
  
  const allImages = tmdbImages.backdrops.concat(data.poster_path);

  return (
    <Container className="w-full flex flex-col gap-8">
      <Box className="flex gap-2 w-full">
        {/** Left hand side, now 1/2 width */}
        <Box className="text-center flex flex-col gap-2 w-1/2">
          <TMDbImages images={allImages} />
          <FilmSidebar data={data} film={film} />
        </Box>

        {/** Right hand side, now 1/2 width */}
        <Box>
          <Paper className="p-4 rounded-lg flex flex-col gap-4">
            <Box className="flex">
              <Typography className="text-left flex-1" variant="h2">{data.title}</Typography>
              <Typography variant="subtitle1">{releaseDate}</Typography>
            </Box>
            <Typography variant="h5" className="text-blue-200">Overview</Typography>
            <Typography>{data.overview}</Typography>
            <Grid container columns={3} spacing={1}>
              {data.genres && data.genres.length > 0 && data.genres.map((genre, index) => (
                <Grid item xs={3} key={index}>
                  <Typography variant="body1">{genre.name}</Typography>
                </Grid>
              ))}
            </Grid>
            <RateFilm />
          </Paper>
        </Box>
      </Box>
      <Box className="flex flex-col items-center m-2">
        <FilmCarousel films={similarToX.recommendations} loading={similarLoading} title={`Because you liked: ${fixMovieLensTitle(similarToX.similar_to.movie_title)}`} />
      </Box>          
    </Container>
  );
}

export default FilmPage;
