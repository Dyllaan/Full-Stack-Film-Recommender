import PropTypes from 'prop-types';
import { Stack, Tooltip, Grid, Accordion, AccordionDetails, AccordionSummary } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import FilmFinances from './FilmFinances';
/**
 * 
 * Content sidebar to pick a content item
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

const FilmSidebar = ({data}) => {

  return (
    <Stack className="mr-2" spacing={2}>
        <Accordion>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel1-content"
            id="panel1-header1"
          >
            Financial Breakdown
          </AccordionSummary>
          <AccordionDetails>
            <FilmFinances revenue={data.revenue} budget={data.budget} />
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel2-content"
            id="panel2-header"
          >
            Languages Spoken
          </AccordionSummary>
          <AccordionDetails>
            <Grid container columns={3} spacing={1}>
              {data.spoken_languages && 
              data.spoken_languages.map((language, index) => (
                <Grid item xs={4} key={index}>
                  <Tooltip title={language.english_name}>
                    <img className="border m-2 mx" src={`https://www.unknown.nu/flags/images/${language.iso_639_1}-100`} width='32' />
                  </Tooltip>
                </Grid>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel3-content"
            id="panel3-header"
          >
            Produced in
          </AccordionSummary>
          <AccordionDetails>
            <Grid container columns={3} spacing={1}>
              {data.production_countries && data.production_countries.map((country, index) => (
                <Tooltip key={index} title={country.name}>
                  <img className="m-2" src={`https://flagsapi.com/${country.iso_3166_1}/flat/64.png`} width='32' />
                </Tooltip>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>
        </Stack>
    );
};

FilmSidebar.propTypes = {
    data: PropTypes.object.isRequired,
    film: PropTypes.object.isRequired,
};

export default FilmSidebar;
